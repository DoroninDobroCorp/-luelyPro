const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');
const lastEl = document.getElementById('last');
const nearSel = document.getElementById('near');
const roomSel = document.getElementById('room');
const startBtn = document.getElementById('start');
const audioEl = document.getElementById('assistantAudio');

// Profile & settings controls
const profileSelect = document.getElementById('profileSelect');
const profileName = document.getElementById('profileName');
const profileNew = document.getElementById('profileNew');
const profileSave = document.getElementById('profileSave');
const profileDelete = document.getElementById('profileDelete');
const voiceSel = document.getElementById('voice');
const langModeSel = document.getElementById('langMode');
const noClarifyChk = document.getElementById('noClarify');
const bulletsMinInp = document.getElementById('bulletsMin');
const bulletsMaxInp = document.getElementById('bulletsMax');
const wordsMinInp = document.getElementById('wordsMin');
const wordsMaxInp = document.getElementById('wordsMax');
const styleTextInp = document.getElementById('styleText');
const thresholdDbInp = document.getElementById('thresholdDb');
const hangMsInp = document.getElementById('hangMs');
const calibrateBtn = document.getElementById('calibrateBtn');
const calStatus = document.getElementById('calStatus');
const shadowEnableChk = document.getElementById('shadowEnable');
const factsFileInp = document.getElementById('factsFile');
const factsTextArea = document.getElementById('factsText');
const factsClearBtn = document.getElementById('factsClear');

let activeProfile = null;

// Local storage helpers
const LS_KEY = 'rc_profiles_v1';
function loadProfiles() {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || '{}'); } catch { return {}; }
}
function saveProfiles(obj) {
  localStorage.setItem(LS_KEY, JSON.stringify(obj));
}
function defaultProfile() {
  return {
    name: 'Default',
    voice: 'marin',
    langMode: 'auto',
    noClarify: true,
    bulletsMin: 1,
    bulletsMax: 2,
    wordsMin: 3,
    wordsMax: 7,
    styleText: 'быстро, уверенно, без вводных',
    thresholdDb: 6,
    hangMs: 250,
    shadowEnable: false,
    factsText: ''
  };
}
function fillForm(p) {
  profileName.value = p.name || '';
  voiceSel.value = p.voice || 'marin';
  langModeSel.value = p.langMode || 'auto';
  noClarifyChk.checked = !!p.noClarify;
  bulletsMinInp.value = p.bulletsMin ?? 1;
  bulletsMaxInp.value = p.bulletsMax ?? 2;
  wordsMinInp.value = p.wordsMin ?? 3;
  wordsMaxInp.value = p.wordsMax ?? 7;
  styleTextInp.value = p.styleText || '';
  thresholdDbInp.value = p.thresholdDb ?? 6;
  hangMsInp.value = p.hangMs ?? 250;
  shadowEnableChk.checked = !!p.shadowEnable;
  factsTextArea.value = p.factsText || '';
}
function readForm() {
  return {
    name: profileName.value || 'Profile',
    voice: voiceSel.value,
    langMode: langModeSel.value,
    noClarify: !!noClarifyChk.checked,
    bulletsMin: Number(bulletsMinInp.value || 1),
    bulletsMax: Number(bulletsMaxInp.value || 2),
    wordsMin: Number(wordsMinInp.value || 3),
    wordsMax: Number(wordsMaxInp.value || 7),
    styleText: styleTextInp.value || '',
    thresholdDb: Number(thresholdDbInp.value || 6),
    hangMs: Number(hangMsInp.value || 250),
    shadowEnable: !!shadowEnableChk.checked,
    factsText: factsTextArea.value || ''
  };
}
function refreshProfileSelect(selectName) {
  const all = loadProfiles();
  const names = Object.keys(all);
  if (names.length === 0) {
    const def = defaultProfile();
    const store = {}; store[def.name] = def; saveProfiles(store);
  }
  const all2 = loadProfiles();
  profileSelect.innerHTML = '';
  Object.keys(all2).forEach(n => {
    const opt = document.createElement('option');
    opt.value = n; opt.textContent = n;
    profileSelect.appendChild(opt);
  });
  if (selectName && all2[selectName]) {
    profileSelect.value = selectName;
  }
  const sel = all2[profileSelect.value] || Object.values(all2)[0];
  activeProfile = sel;
  fillForm(sel);
}

// Simple CSV/JSON parser -> text block
async function readFactsFile(file) {
  if (!file) return '';
  const text = await file.text();
  try {
    const obj = JSON.parse(text);
    return JSON.stringify(obj).slice(0, 2000);
  } catch {
    // CSV/TXT fallback: return first 2000 chars
    return text.slice(0, 2000);
  }
}

// Build instruction snippet from profile
function buildStyleInstructions(p) {
  const lang = p.langMode === 'ru' ? 'Отвечай по-русски.' : (p.langMode === 'en' ? 'Answer in English.' : 'Язык — как у вопроса.');
  const clar = p.noClarify ? 'Не задавай уточняющих вопросов.' : 'Если данных не хватает — один уточняющий вопрос.';
  const bullets = `Формат: ${p.bulletsMin}–${p.bulletsMax} буллета(ов), по ${p.wordsMin}–${p.wordsMax} слов.`;
  const style = p.styleText ? `Стиль: ${p.styleText}.` : '';
  return `${lang} ${clar} ${bullets} ${style}`.trim();
}
function buildStructureHint() {
  return 'Структура: стратегический — контекст/рынок → метрика/факт → план/риски; финансовый — юнит-экономика (CAC/LTV) → драйверы → контроль рисков; технический — архитектура → масштабирование → надёжность.';
}
function compactText(t, maxWords = 20) {
  const words = (t || '').trim().split(/\s+/).filter(Boolean);
  return words.slice(-maxWords).join(' ');
}

function log(...a){ logEl.textContent += a.join(' ') + '\n'; logEl.scrollTop = logEl.scrollHeight; }

async function listDevices() {
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true }); // prompt permission to reveal labels
    const devs = await navigator.mediaDevices.enumerateDevices();
    const mics = devs.filter(d => d.kind === 'audioinput');
    for (const s of [nearSel, roomSel]) { s.innerHTML = ''; }
    mics.forEach(d => {
      const opt1 = document.createElement('option');
      opt1.value = d.deviceId; opt1.textContent = d.label || d.deviceId;
      nearSel.appendChild(opt1.cloneNode(true));
      roomSel.appendChild(opt1);
    });
  } catch (e) {
    log('Device access error:', e?.message || e);
  }
}

listDevices();

// Initialize profiles UI
refreshProfileSelect();

profileSelect.onchange = () => {
  const all = loadProfiles();
  const p = all[profileSelect.value];
  if (p) { activeProfile = p; fillForm(p); }
};
profileNew.onclick = () => {
  const p = defaultProfile();
  p.name = `Profile ${Math.floor(Math.random()*1000)}`;
  activeProfile = p; fillForm(p);
};
profileSave.onclick = () => {
  const p = readForm();
  const store = loadProfiles();
  // If renaming, delete old key
  const oldName = profileSelect.value;
  if (oldName && oldName !== p.name && store[oldName]) delete store[oldName];
  store[p.name] = p; saveProfiles(store);
  refreshProfileSelect(p.name);
  log('Профиль сохранён:', p.name);
};
profileDelete.onclick = () => {
  const sel = profileSelect.value;
  const store = loadProfiles();
  if (store[sel]) { delete store[sel]; saveProfiles(store); }
  refreshProfileSelect();
  log('Профиль удалён:', sel);
};
factsFileInp.onchange = async (e) => {
  const file = e.target.files?.[0];
  const txt = await readFactsFile(file);
  factsTextArea.value = txt;
};
factsClearBtn.onclick = () => { factsTextArea.value = ''; };

// Calibration: 10s silence baseline
calibrateBtn.onclick = async () => {
  try {
    const nearId = nearSel.value || undefined;
    const roomId = roomSel.value || undefined;
    calStatus.textContent = 'калибровка 10с…';
    const { threshold } = await calibrateGate(nearId, roomId, (left) => { calStatus.textContent = `калибровка… ${left}s`; });
    thresholdDbInp.value = String(threshold.toFixed(1));
    calStatus.textContent = `ок, порог ≈ ${threshold.toFixed(1)} dB`;
  } catch (e) {
    calStatus.textContent = 'ошибка калибровки';
    log('Calibrate error:', e?.message || e);
  }
};

async function calibrateGate(nearDeviceId, roomDeviceId, tickCb) {
  const nearStream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: nearDeviceId ? { exact: nearDeviceId } : undefined, echoCancellation: true, noiseSuppression: true } });
  const roomStream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: roomDeviceId ? { exact: roomDeviceId } : undefined, echoCancellation: true, noiseSuppression: true } });
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
  const nearNode = ctx.createMediaStreamSource(nearStream);
  const roomNode = ctx.createMediaStreamSource(roomStream);
  const aNear = ctx.createAnalyser(); aNear.fftSize = 1024; aNear.smoothingTimeConstant = 0.8;
  const aRoom = ctx.createAnalyser(); aRoom.fftSize = 1024; aRoom.smoothingTimeConstant = 0.8;
  nearNode.connect(aNear); roomNode.connect(aRoom);
  const bufN = new Float32Array(aNear.fftSize);
  const bufR = new Float32Array(aRoom.fftSize);
  let sumNear = 0, sumRoom = 0, count = 0;
  const duration = 10; // seconds
  for (let s = duration; s > 0; s--) {
    tickCb?.(s);
    const start = performance.now();
    while (performance.now() - start < 1000) {
      aNear.getFloatTimeDomainData(bufN);
      aRoom.getFloatTimeDomainData(bufR);
      const dbN = rmsToDb(bufN);
      const dbR = rmsToDb(bufR);
      sumNear += dbN; sumRoom += dbR; count++;
      await new Promise(r => setTimeout(r, 100));
    }
  }
  const meanNear = sumNear / Math.max(1, count);
  const meanRoom = sumRoom / Math.max(1, count);
  const delta = meanNear - meanRoom; // how much near is inherently louder
  const threshold = Math.max(6, delta + 6); // baseline delta + margin
  // cleanup
  nearStream.getTracks().forEach(t => t.stop());
  roomStream.getTracks().forEach(t => t.stop());
  await ctx.close();
  return { threshold, meanNear, meanRoom, delta };
}
function rmsToDb(arr) {
  let sum = 0; for (let i=0;i<arr.length;i++){ const v = arr[i]; sum += v*v; }
  const rms = Math.sqrt(sum / Math.max(1, arr.length));
  return 20 * Math.log10(rms + 1e-9);
}

startBtn.onclick = async () => {
  try {
    startBtn.disabled = true;
    statusEl.textContent = 'starting…';
    // Freeze current profile
    activeProfile = readForm();

    // 1) Capture two inputs
    const nearStream = await navigator.mediaDevices.getUserMedia({
      audio: { deviceId: nearSel.value ? { exact: nearSel.value } : undefined, echoCancellation: true, noiseSuppression: true }
    });
    const roomStream = await navigator.mediaDevices.getUserMedia({
      audio: { deviceId: roomSel.value ? { exact: roomSel.value } : undefined, echoCancellation: true, noiseSuppression: true }
    });

    // 2) Worklet gate (2 -> 1)
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    await ctx.audioWorklet.addModule('gating-worklet.js');
    const nearNode = ctx.createMediaStreamSource(nearStream);
    const roomNode = ctx.createMediaStreamSource(roomStream);
    const gate = new AudioWorkletNode(
      ctx,
      'gate-processor',
      {
        numberOfInputs: 2,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: 'explicit',
        processorOptions: { thresholdDb: Number(thresholdDbInp.value||6), hangMs: Number(hangMsInp.value||250) }
      }
    );
    nearNode.connect(gate, 0, 0); // input[0]
    roomNode.connect(gate, 0, 1); // input[1]

    const mixDest = ctx.createMediaStreamDestination();
    gate.connect(mixDest);

    // status log
    gate.port.onmessage = (e) => {
      const { state, dbNear, dbRoom } = e.data || {}; if (!state) return;
      statusEl.textContent = `stream: ${state}  near:${dbNear.toFixed(1)}dB room:${dbRoom.toFixed(1)}dB`;
    };

    // 3) WebRTC -> OpenAI Realtime
    const sess = await (await fetch('/session')).json();
    if (!sess?.client_secret?.value) { throw new Error('No client_secret from server'); }
    const { client_secret } = sess;

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    // send processed stream (only audience)
    pc.addTrack(mixDest.stream.getAudioTracks()[0], mixDest.stream);

    // receive assistant audio
    pc.ontrack = (e) => { audioEl.srcObject = e.streams[0]; };

    // datachannel for events
    const dc = pc.createDataChannel('oai-events');
    dc.onmessage = (ev) => {
      try {
        const evt = JSON.parse(ev.data);
        handleServerEvent(evt);
      } catch (err) {
        log('DC parse error:', err?.message || err);
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const sdpResp = await fetch('https://api.openai.com/v1/realtime?model=gpt-realtime', {
      method: 'POST',
      body: offer.sdp,
      headers: {
        Authorization: `Bearer ${client_secret.value}`,
        'Content-Type': 'application/sdp',
        'OpenAI-Beta': 'realtime=v1'
      }
    });

    if (!sdpResp.ok) {
      const txt = await sdpResp.text();
      throw new Error(`Realtime SDP error ${sdpResp.status}: ${txt}`);
    }

    const answer = { type: 'answer', sdp: await sdpResp.text() };
    await pc.setRemoteDescription(answer);
    log('WebRTC connected');

    // 4) Configure session after DC opens
    dc.onopen = () => {
      send({ type: 'session.update', session: {
        input_audio_transcription: { model: 'gpt-4o-mini-transcribe' },
        voice: activeProfile.voice,
        instructions: `Ты — незаметный деловой коуч в ухе. Отвечай ТОЛЬКО по явному TRIGGER от клиента. ${buildStyleInstructions(activeProfile)}`
      }});
    };

    function send(obj){ dc.send(JSON.stringify(obj)); }

    // server event handler — collect transcript and fire TRIGGER
    let lastUserText = '';
    let shadowSummary = '';
    if (activeProfile.shadowEnable) {
      // Start shadow transcription of near mic
      try { await startShadowSession(nearStream, (s) => { shadowSummary = s; }); } catch (e) { log('Shadow error:', e?.message || e); }
    }
    function handleServerEvent(evt){
      // conversation.item.created may carry the user transcript
      if (evt.type === 'conversation.item.created') {
        const item = evt.item || {};
        if (item.role === 'user') {
          const text = (item?.content || [])
            .filter(c => c.type === 'input_text' || c.type === 'output_text' || c.type === 'transcript')
            .map(c => c.text || c.transcript || '')
            .join(' ')
            .trim();
          if (text) { lastUserText = text; lastEl.textContent = text; }
        }
      }

      if (evt.type === 'input_audio_buffer.speech_stopped') {
        if (isAudienceQuestion(lastUserText)) {
          triggerCoach(lastUserText);
        }
      }

      if (evt.type === 'error') { log('ERROR', evt.error || ''); }
    }

    function isAudienceQuestion(text){
      if (!text) return false;
      const t = text.toLowerCase();
      const Q = [
        '?','почему','зачем','как','когда','сколько','можете','не могли бы','что если','каков','объясните','расскажите','уточните',
        'where','when','why','how','what','which','could you','can you','would you','explain','clarify'
      ];
      return Q.some(k => t.includes(k));
    }

    function triggerCoach(questionText){
      const style = buildStyleInstructions(activeProfile);
      const structure = buildStructureHint();
      const facts = (activeProfile.factsText || factsTextArea.value || '').trim();
      const factsBlock = facts ? `Контекст фактов: ${facts.slice(0, 1500)}.` : '';
      const shadowBlock = shadowSummary ? `Краткий конспект докладчика: ${compactText(shadowSummary, 30)}.` : '';

      send({ type: 'response.create', response: {
        instructions: `Вопрос аудитории: "${questionText}". ${style} ${structure} ${factsBlock} ${shadowBlock}`.trim(),
        modalities: ['audio']
      }});
    }
  } catch (e) {
    log('Start error:', e?.message || e);
    startBtn.disabled = false;
    statusEl.textContent = 'idle';
  }
};

// Shadow session: returns a Promise that resolves to rolling summary string producer
async function startShadowSession(nearStream, onSummary) {
  // independent session for near mic transcription
  const sess = await (await fetch('/session')).json();
  if (!sess?.client_secret?.value) throw new Error('No client_secret (shadow)');
  const { client_secret } = sess;

  const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
  pc.addTrack(nearStream.getAudioTracks()[0], nearStream);
  let summary = '';
  let lastNearText = '';
  const dc = pc.createDataChannel('oai-events');
  dc.onmessage = (ev) => {
    try {
      const evt = JSON.parse(ev.data);
      if (evt.type === 'conversation.item.created') {
        const item = evt.item || {};
        if (item.role === 'user') {
          const text = (item?.content || [])
            .filter(c => c.type === 'input_text' || c.type === 'transcript' || c.type === 'output_text')
            .map(c => c.text || c.transcript || '')
            .join(' ')
            .trim();
          if (text) lastNearText = text;
        }
      }
      if (evt.type === 'input_audio_buffer.speech_stopped') {
        if (lastNearText) {
          summary = compactText(lastNearText, 30);
          try { onSummary?.(summary); } catch {}
        }
      }
    } catch {}
  };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const sdpResp = await fetch('https://api.openai.com/v1/realtime?model=gpt-realtime', {
    method: 'POST',
    body: offer.sdp,
    headers: {
      Authorization: `Bearer ${client_secret.value}`,
      'Content-Type': 'application/sdp',
      'OpenAI-Beta': 'realtime=v1'
    }
  });
  if (!sdpResp.ok) throw new Error(`Shadow SDP error ${sdpResp.status}`);
  const answer = { type: 'answer', sdp: await sdpResp.text() };
  await pc.setRemoteDescription(answer);
  dc.onopen = () => {
    dc.send(JSON.stringify({ type: 'session.update', session: {
      input_audio_transcription: { model: 'gpt-4o-mini-transcribe' },
      instructions: 'Тихий теневой канал: не отвечай и не говори. Только транскрибируй вход и ничего не произноси.'
    }}));
  };
  // Await initial establishment then return
  Object.defineProperty(window, '_shadowSummary', { get(){ return summary; } });
  return;
}
