/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import {GoogleGenAI, LiveServerMessage, Modality, Blob, Type} from '@google/genai';

// --- UI Elements ---
const connectButton = document.getElementById('connect-button') as HTMLButtonElement;
const statusDiv = document.getElementById('status') as HTMLParagraphElement;
const conversationControls = document.getElementById('conversation-controls') as HTMLDivElement;
const displayAreas = document.getElementById('display-areas') as HTMLDivElement;
const userMicButton = document.getElementById('user-mic-button') as HTMLButtonElement;
const interlocutorMicButton = document.getElementById('interlocutor-mic-button') as HTMLButtonElement;
const thesesDisplay = document.getElementById('theses-display') as HTMLDivElement;
const transcriptionDisplay = document.getElementById('transcription-display') as HTMLDivElement;

// --- State ---
let isConnected = false;
type Speaker = 'user' | 'interlocutor' | null;
let currentSpeaker: Speaker = null;
let sessionPromise: Promise<any> | null = null;
let stream: MediaStream | null = null;
let inputAudioContext: AudioContext | null = null;
let scriptProcessor: ScriptProcessorNode | null = null;
let sourceNode: MediaStreamAudioSourceNode | null = null;
let currentTranscriptionNode: Text | null = null;

let pythonBackendSocket: WebSocket | null = null;

// --- App Logic ---
const ai = new GoogleGenAI({apiKey: process.env.API_KEY});

connectButton.addEventListener('click', () => {
  if (isConnected) {
    disconnect();
  } else {
    connect();
  }
});

userMicButton.addEventListener('click', () => setSpeaker('user'));
interlocutorMicButton.addEventListener('click', () => setSpeaker('interlocutor'));

function setStatus(text: string) {
  statusDiv.textContent = text;
}

function setSpeaker(speaker: Speaker) {
    if (!isConnected || currentSpeaker === speaker) return;

    currentSpeaker = speaker;
    
    // Notify Python backend about the speaker change
    if (pythonBackendSocket && pythonBackendSocket.readyState === WebSocket.OPEN) {
        pythonBackendSocket.send(JSON.stringify({
            type: 'speaker_change',
            speaker: currentSpeaker
        }));
    }
    
    // Add a label to the transcription log instead of clearing it
    if (currentSpeaker) {
        const speakerLabel = document.createElement('p');
        speakerLabel.className = 'speaker-label';
        speakerLabel.textContent = `--- ${speaker.charAt(0).toUpperCase() + speaker.slice(1)} Speaking ---`;
        transcriptionDisplay.appendChild(speakerLabel);
        transcriptionDisplay.scrollTop = transcriptionDisplay.scrollHeight;
        currentTranscriptionNode = null; // Start a new text node for the new speaker
    }
    
    updateUI();
}

function updateUI() {
    userMicButton.classList.toggle('active', currentSpeaker === 'user');
    interlocutorMicButton.classList.toggle('active', currentSpeaker === 'interlocutor');
    if (currentSpeaker) {
        setStatus(`Listening to: ${currentSpeaker}`);
    }
}

async function connect() {
  isConnected = true;
  connectButton.textContent = 'Disconnect';
  connectButton.classList.add('listening');
  setStatus('Connecting to Python & Gemini...');
  conversationControls.style.display = 'block';
  displayAreas.style.display = 'grid';
  transcriptionDisplay.innerHTML = '<p class="placeholder">Live transcription will be shown here.</p>';


  // 1. Connect to Python Backend
  try {
    pythonBackendSocket = new WebSocket('ws://localhost:8765');
    
    pythonBackendSocket.onopen = () => {
        setStatus('Connected to Python. Connecting to Gemini...');
    };

    pythonBackendSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.command === 'speak' && data.text) {
            speakText(data.text);
        }
        // You can add more commands from Python here, e.g., to update thesesDisplay
    };

    pythonBackendSocket.onerror = (error) => {
        console.error('WebSocket Error:', error);
        setStatus('Error: Could not connect to Python backend. Is it running?');
        disconnect();
    };

    pythonBackendSocket.onclose = () => {
        if (isConnected) { // Prevent status update on manual disconnect
            setStatus('Connection to Python backend lost.');
            disconnect();
        }
    };

  } catch (e) {
      console.error("Failed to create WebSocket", e);
      setStatus('Error: Failed to initialize connection to Python backend.');
      disconnect();
      return;
  }

  // 2. Connect to Gemini API
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: true});
    inputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({sampleRate: 16000});
    
    sessionPromise = ai.live.connect({
      model: 'gemini-2.5-flash-native-audio-preview-09-2025',
      config: {
        responseModalities: [Modality.AUDIO],
        inputAudioTranscription: {},
      },
      callbacks: {
        onopen: () => {
          setStatus('Connected. Select who is speaking.');
          sourceNode = inputAudioContext!.createMediaStreamSource(stream!);
          scriptProcessor = inputAudioContext!.createScriptProcessor(4096, 1, 1);
          
          scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
            if (!currentSpeaker) return; // Don't send audio if no speaker is selected
            const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
            const pcmBlob = createBlob(inputData);
            sessionPromise!.then((session) => {
              session.sendRealtimeInput({media: pcmBlob});
            });
          };

          sourceNode.connect(scriptProcessor);
          scriptProcessor.connect(inputAudioContext!.destination);
        },
        onmessage: async (message: LiveServerMessage) => {
           if (message.serverContent?.inputTranscription) {
                const text = message.serverContent.inputTranscription.text;
                updateTranscriptionDisplay(text);
                // Forward transcription to Python backend
                if (pythonBackendSocket && pythonBackendSocket.readyState === WebSocket.OPEN) {
                    pythonBackendSocket.send(JSON.stringify({
                        type: 'transcription',
                        text: text
                    }));
                }
           }
        },
        onerror: (e: ErrorEvent) => {
          console.error('Gemini Error:', e);
          setStatus(`Gemini Error: ${e.message}`);
          disconnect();
        },
        onclose: () => {
          if (isConnected) { disconnect(); }
        },
      },
    });

    await sessionPromise;

  } catch (error) {
    console.error('Failed to connect to Gemini:', error);
    setStatus(`Gemini Error: ${(error as Error).message}`);
    disconnect();
  }
}

function disconnect() {
  if (!isConnected) return;
  
  isConnected = false;
  currentSpeaker = null;
  connectButton.textContent = 'Connect';
  connectButton.classList.remove('listening');
  setStatus('Disconnected');
  conversationControls.style.display = 'none';
  displayAreas.style.display = 'none';
  userMicButton.classList.remove('active');
  interlocutorMicButton.classList.remove('active');

  if (pythonBackendSocket) {
      pythonBackendSocket.close();
      pythonBackendSocket = null;
  }
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (inputAudioContext) {
    inputAudioContext.close();
    inputAudioContext = null;
  }
  if (sessionPromise) {
    sessionPromise.then(session => session.close());
    sessionPromise = null;
  }
}

function updateTranscriptionDisplay(text: string) {
    if (transcriptionDisplay.querySelector('.placeholder')) {
        transcriptionDisplay.innerHTML = '';
    }

    if (!currentTranscriptionNode) {
        currentTranscriptionNode = document.createTextNode(text);
        const p = document.createElement('p');
        p.className = 'transcription-line';
        p.appendChild(currentTranscriptionNode);
        transcriptionDisplay.appendChild(p);
    } else {
        currentTranscriptionNode.nodeValue += text;
    }

    transcriptionDisplay.scrollTop = transcriptionDisplay.scrollHeight;
}

function speakText(text: string) {
    if (text.length > 0 && 'speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }
}

thesesDisplay.innerHTML = '<p class="placeholder">Theses generated by your Python backend will be handled via audio prompts.</p>';


// --- Audio Utility Functions ---
function createBlob(data: Float32Array): Blob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] < 0 ? data[i] * 32768 : data[i] * 32767;
  }
  return { data: encode(new Uint8Array(int16.buffer)), mimeType: 'audio/pcm;rate=16000' };
}

function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) { binary += String.fromCharCode(bytes[i]); }
  return btoa(binary);
}