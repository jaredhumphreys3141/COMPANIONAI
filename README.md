# CompanionAI

A local AI companion you build yourself: it listens, talks back, and can draw
itself. Everything runs on the machine in front of you — a PC, a Raspberry Pi 5
or an NVIDIA Jetson. There is no account, no API key and no cloud, and the app
starts with the network switched off every single time.

```
   speech in  ──▶  Whisper (faster-whisper)  ──▶  ┐
                                                  ├──▶  llama.cpp  ──▶  Piper  ──▶  speech out
   keyboard   ─────────────────────────────────▶  ┘         │
                                                            └──▶  Stable Diffusion ──▶ pictures
```

## What it does

- **Talk to it.** Type, push-to-talk, or leave it hands-free: it listens with
  voice-activity detection, answers out loud, and lets you interrupt mid-sentence.
- **Build a companion from the GUI.** Name, personality, manner, voice, art
  style — plus every model choice and sampling parameter, with defaults already
  tuned for the board you are on. Companions are one JSON file each, so you can
  copy them between machines.
- **See it.** Generate a reference portrait and any scene you like, with the
  character's face seed kept fixed so it stays recognisable.
- **Stay offline.** The network is blocked by default. Downloading a model is
  the only thing that ever needs it, it requires an explicit approval in the
  GUI, and every attempt — allowed or blocked — is written to an activity log
  you can read on the Settings tab.

## Install

```bash
git clone https://github.com/jaredhumphreys3141/COMPANIONAI.git
cd COMPANIONAI
./scripts/install.sh          # detects your board and installs what fits it
```

The installer creates `.venv`, pulls in the system packages it needs
(PortAudio, espeak-ng, a compiler), and on a CUDA machine builds
`llama-cpp-python` with GPU offload. See [docs/hardware.md](docs/hardware.md)
for board-specific notes.

## Run

```bash
./scripts/run.sh              # then open http://127.0.0.1:7860
```

Useful flags:

| command | what it does |
| --- | --- |
| `companionai --doctor` | what is installed, what is missing, what is ready |
| `companionai --smoke-test --allow-network` | prove the real models work end to end |
| `companionai --list-devices` | microphones and speakers the app can see |
| `companionai --host 0.0.0.0` | reachable from other devices on your LAN |
| `companionai --home /mnt/ssd/companion` | keep models and settings elsewhere |

## First five minutes

1. **Settings → Network** — choose *Allow for this session only* and press Apply.
2. **Models** — install one of each: a language model, a speech-recognition
   model, a voice, and (optionally) an image model. The `suits this device`
   column tells you which ones will be comfortable on your hardware.
3. **Settings → Network** — set it back to *Offline*. Nothing below needs it.
4. **Studio** — write a persona, pick the manner, press *Save and make active*.
   *Generate reference portrait* gives the character a face.
5. **Talk** — press *Start listening* and just talk.

## What runs where

| | PC (CPU) | PC / Jetson (CUDA) | Raspberry Pi 5 |
| --- | --- | --- | --- |
| speech recognition | ~1s | real time | 1–3s |
| reply (first words) | 1–3s | under 1s | 2–5s |
| speech synthesis | faster than real time | faster than real time | faster than real time |
| a 512px image | ~40s | 1–5s | several minutes |

The Jetson is the target for a genuinely conversational experience: the whole
model stack sits on the iGPU. On a PC without a GPU and on the Pi you get the
same features with a pause between turns.

## Privacy

- Offline is the default and a session approval never survives a restart.
- Approving network access only permits **you** to download models. Conversations,
  audio, transcripts and images are never uploaded, whatever the setting.
- Telemetry in the underlying libraries (Gradio, HuggingFace) is switched off
  before they are imported.
- `--share` (a public Gradio tunnel) is off unless you ask for it, and warns first.
- The audit log lives at `~/.companionai/logs/network-audit.jsonl`.
- The interface can require a password (Settings → Runtime → Sign-in). Set one
  before binding to anything other than `127.0.0.1`; only a salted hash is stored.

More detail in [docs/privacy.md](docs/privacy.md).

## Built on

[Gradio](https://gradio.app) for the interface, [llama.cpp](https://github.com/ggerganov/llama.cpp)
for language models, [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
for speech recognition, [Piper](https://github.com/OHF-Voice/piper1-gpl) for
speech, and [Diffusers](https://github.com/huggingface/diffusers) for images.
CompanionAI is the glue: model management, the offline gate, the character
system and the conversation pipeline.

## Layout

```
companionai/
  hardware.py      board detection and per-board defaults
  net.py           the network gate and its audit log
  catalog.py       model catalogue, install/remove
  character.py     a companion and all its parameters
  llm.py           llama.cpp / Ollama backends
  asr.py           faster-whisper
  tts.py           Piper (espeak-ng fallback) and playback
  imagegen.py      Diffusers
  voice.py         microphone loop, VAD, barge-in
  conversation.py  history and the streaming reply pipeline
  session.py       the running companion
  ui/app.py        the Gradio interface
```

## Licence

MIT — see [LICENSE](LICENSE). Models you download carry their own licences,
shown in the Models tab.
