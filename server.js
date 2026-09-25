import express from 'express';
import multer from 'multer';
import fs from 'node:fs';
import path from 'node:path';
import {
  loadModel,
  completion,
  textToSpeech,
  SMOLVLM2_500M_MULTIMODAL_Q8_0,
  MMPROJ_SMOLVLM2_500M_MULTIMODAL_Q8_0,
  TTS_MULTILINGUAL_SUPERTONIC3_Q8_0,
} from '@qvac/sdk';

fs.mkdirSync('./uploads', { recursive: true });
fs.mkdirSync('./output',  { recursive: true });

const upload = multer({ dest: './uploads/' });
const app = express();
app.use(express.static('public'));
app.use('/audio', express.static('output'));

function writeWav(filePath, samples, sampleRate) {
  const n = samples.length;
  const buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + n * 2, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    buf.writeInt16LE(Math.max(-32768, Math.min(32767, samples[i])), 44 + i * 2);
  }
  fs.writeFileSync(filePath, buf);
}

let visionId, ttsId;
const PROMPT =
  'Look at this image carefully. First, in one short phrase, name the main object. ' +
  'Then, as that object, write a dramatic first-person confession (3-5 sentences) ' +
  'about what it has witnessed, resented, or secretly dreamed of. ' +
  'Be specific, dramatic, and slightly petty. Output format exactly:\n' +
  'OBJECT: <name>\n' +
  'CONFESSION: <text>';

async function boot() {
  console.log('→ Loading vision model...');
  visionId = await loadModel({
    modelSrc: SMOLVLM2_500M_MULTIMODAL_Q8_0,
    modelType: 'llm',
    modelConfig: {
      projectionModelSrc: MMPROJ_SMOLVLM2_500M_MULTIMODAL_Q8_0,
      ctx_size: 2048,
    },
  });

  console.log('→ Loading TTS voice...');
  ttsId = await loadModel({
    modelSrc: TTS_MULTILINGUAL_SUPERTONIC3_Q8_0.src,
    modelType: 'tts',
    modelConfig: { ttsEngine: 'supertonic', language: 'en' },
  });

  console.log('✅ Models ready.');
}

app.post('/api/confess', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image uploaded.' });
  const imagePath = req.file.path;

  try {
    const run = completion({
      modelId: visionId,
      history: [
        { role: 'user', content: PROMPT, attachments: [{ path: imagePath }] },
      ],
      stream: true,
    });
    let raw = '';
    for await (const token of run.tokenStream) raw += token;

    const match = raw.match(/CONFESSION:\s*([\s\S]+)/i);
    const confession = (match ? match[1] : raw).trim();
    if (!confession) throw new Error('No confession generated.');

    const audio = await textToSpeech({
      modelId: ttsId,
      text: confession,
      inputType: 'text',
      stream: false,
    });

    const samples = await audio.buffer;
    const sampleRate = (await audio.sampleRate) || 44100;

    if (!Array.isArray(samples) || samples.length === 0) {
      throw new Error(
        'TTS returned no samples. Raw: ' + JSON.stringify(audio).slice(0, 300)
      );
    }

    const wavName = `confession-${Date.now()}.wav`;
    const wavPath = path.resolve('./output', wavName);
    writeWav(wavPath, samples, sampleRate);

    res.json({ confession, audioUrl: `/audio/${wavName}` });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err?.message ?? String(err) });
  } finally {
    fs.unlink(imagePath, () => {});
  }
});

boot()
  .then(() => {
    app.listen(3000, () => {
      console.log('\n🕯️   Open http://localhost:3000\n');
    });
  })
  .catch((err) => {
    console.error('Boot failed:', err);
    process.exit(1);
  });
