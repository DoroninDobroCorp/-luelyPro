// Realtime Coach MVP server
// ENV:
//   OPENAI_API_KEY=sk-...
//   PORT=3000 (optional)

import 'dotenv/config';
import express from 'express';

const app = express();
app.use(express.static('public'));

app.get('/session', async (req, res) => {
  try {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      res.status(500).json({ error: 'OPENAI_API_KEY is not set' });
      return;
    }

    const r = await fetch('https://api.openai.com/v1/realtime/client_secrets', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'OpenAI-Beta': 'realtime=v1'
      },
      body: JSON.stringify({
        session: {
          type: 'realtime',
          model: 'gpt-realtime',
          // speech↔speech
          modalities: ['audio'],
          voice: 'marin', // e.g., 'marin', 'alloy', 'cedar'
          instructions: `Ты — незаметный деловой коуч в ухе. Говори ТОЛЬКО когда получишь явную команду TRIGGER от клиента. Никогда не начинай первым. Формат ответа: до двух буллетов по 3–7 слов каждый, затем мягкая связка (1 короткая фраза). Стиль: уверенно, без воды, без извинений. Язык — как у вопроса. Если не хватает данных — предложи уточнение одним коротким вопросом. Не комментируй речь докладчика; отвечай на СМЫСЛ вопроса аудитории и помогай строить каркас ответа (рынок → метрики → риски → план).`,
          // server-side transcription of the incoming audio
          input_audio_transcription: {
            model: 'gpt-4o-mini-transcribe'
          }
          // turn_detection: default server VAD
        }
      })
    });

    if (!r.ok) {
      const text = await r.text();
      console.error('Failed to mint client secret:', r.status, text);
      res.status(500).json({ error: 'failed to create client secret' });
      return;
    }

    const data = await r.json();
    res.json(data);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'failed to create client secret' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Realtime Coach server on http://localhost:${PORT}`));
