// Hybrid client: remote mic/ear for Python backend
(function () {
  const connectButton = document.getElementById('connect-button');
  const statusDiv = document.getElementById('status');

  let ws = null;
  let stream = null;
  let audioCtx = null;
  let sourceNode = null;
  let processor = null;

  // Buffers for resampling to 16kHz
  let inputSampleRate = 48000;
  let floatBuffer = new Float32Array(0);
  const TARGET_SR = 16000;
  const FRAME_SAMPLES = 320; // 20ms @ 16kHz
  let resampledBuffer = new Float32Array(0);

  function setStatus(msg) {
    statusDiv.textContent = msg;
  }

  connectButton.addEventListener('click', () => {
    if (ws) {
      disconnect();
    } else {
      connect();
    }
  });

  async function connect() {
    try {
      setStatus('Connecting...');
      ws = new WebSocket('ws://127.0.0.1:8765');
      ws.binaryType = 'arraybuffer';

      ws.onopen = async () => {
        setStatus('Connected. Initializing audio...');
        connectButton.textContent = 'Disconnect';
        connectButton.classList.add('listening');
        try {
          await startAudio();
          setStatus('Streaming audio to backend...');
        } catch (e) {
          console.error(e);
          setStatus('Audio init failed: ' + (e && e.message ? e.message : e));
          disconnect();
        }
      };

      ws.onmessage = async (ev) => {
        // Binary WAV from server
        try {
          const arrayBuffer = ev.data; // ArrayBuffer
          if (!(arrayBuffer && arrayBuffer.byteLength)) return;
          if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          }
          const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
          const src = audioCtx.createBufferSource();
          src.buffer = audioBuffer;
          src.connect(audioCtx.destination);
          src.start();
        } catch (e) {
          console.error('Playback error:', e);
        }
      };

      ws.onerror = (e) => {
        console.error('WS error', e);
        setStatus('WebSocket error. Is backend running?');
      };

      ws.onclose = () => {
        setStatus('Disconnected');
        cleanupAudio();
        ws = null;
        connectButton.textContent = 'Connect';
        connectButton.classList.remove('listening');
      };
    } catch (e) {
      console.error(e);
      setStatus('Connection failed');
      ws = null;
    }
  }

  function disconnect() {
    if (ws) {
      try { ws.close(); } catch {}
      ws = null;
    }
    cleanupAudio();
    connectButton.textContent = 'Connect';
    connectButton.classList.remove('listening');
    setStatus('Disconnected');
  }

  async function startAudio() {
    // Capture mic with browser AEC/NS/AGC
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 48000,
        sampleSize: 16,
      },
      video: false,
    });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    inputSampleRate = audioCtx.sampleRate || 16000;

    sourceNode = audioCtx.createMediaStreamSource(stream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      appendFloatBuffer(input);
      // Resample accumulated float buffer to 16k and send in 20ms frames
      resampleAndSend();
    };

    sourceNode.connect(processor);
    processor.connect(audioCtx.destination);
  }

  function cleanupAudio() {
    try { if (processor) { processor.disconnect(); processor.onaudioprocess = null; } } catch {}
    try { if (sourceNode) sourceNode.disconnect(); } catch {}
    try { if (audioCtx) audioCtx.close(); } catch {}
    if (stream) {
      for (const t of stream.getTracks()) try { t.stop(); } catch {}
    }
    stream = null; audioCtx = null; sourceNode = null; processor = null;
    floatBuffer = new Float32Array(0);
    resampledBuffer = new Float32Array(0);
  }

  function appendFloatBuffer(chunk) {
    // Concatenate chunk to floatBuffer
    const tmp = new Float32Array(floatBuffer.length + chunk.length);
    tmp.set(floatBuffer, 0);
    tmp.set(chunk, floatBuffer.length);
    floatBuffer = tmp;
  }

  function resampleAndSend() {
    if (floatBuffer.length === 0) return;

    let out; // Float32 at 16k
    if (inputSampleRate === TARGET_SR) {
      out = floatBuffer;
      floatBuffer = new Float32Array(0);
    } else {
      out = resampleLinear(floatBuffer, inputSampleRate, TARGET_SR);
      floatBuffer = new Float32Array(0);
    }

    // Append to resampledBuffer
    if (resampledBuffer.length === 0) {
      resampledBuffer = out;
    } else {
      const tmp = new Float32Array(resampledBuffer.length + out.length);
      tmp.set(resampledBuffer, 0);
      tmp.set(out, resampledBuffer.length);
      resampledBuffer = tmp;
    }

    // While we have at least 20ms frame (320 samples), send
    let offset = 0;
    while (resampledBuffer.length - offset >= FRAME_SAMPLES) {
      const frame = resampledBuffer.subarray(offset, offset + FRAME_SAMPLES);
      offset += FRAME_SAMPLES;
      const pcm = floatToPCM16(frame);
      try {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(pcm.buffer);
      } catch {}
    }
    if (offset > 0) {
      // keep leftover
      const remaining = resampledBuffer.length - offset;
      const tmp = new Float32Array(remaining);
      tmp.set(resampledBuffer.subarray(offset));
      resampledBuffer = tmp;
    }
  }

  function resampleLinear(input, fromRate, toRate) {
    if (fromRate === toRate) return input.slice();
    const ratio = fromRate / toRate;
    const newLen = Math.floor(input.length / ratio);
    const out = new Float32Array(newLen);
    let pos = 0;
    for (let i = 0; i < newLen; i++) {
      const idx = i * ratio;
      const i0 = Math.floor(idx);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const frac = idx - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  function floatToPCM16(f32) {
    const out = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      let s = Math.max(-1, Math.min(1, f32[i]));
      out[i] = s < 0 ? s * 32768 : s * 32767;
    }
    return out;
  }
})();
