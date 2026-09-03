# Board notes

CompanionAI detects the machine at start-up and picks defaults for it. This
page covers what the installer cannot do for you.

## Raspberry Pi 5

Works well, with a pause between turns. 8 GB is comfortable; 4 GB works with a
1.5B model and no image generation.

```bash
sudo apt install python3-venv python3-dev build-essential cmake \
                 libportaudio2 portaudio19-dev libsndfile1 espeak-ng
./scripts/install.sh --minimal
```

Recommended models: **Qwen2.5 1.5B Q4_K_M**, **Whisper tiny.en**, **Piper
en_US-amy-low**.

Notes:

- Put `COMPANIONAI_HOME` on an SSD or a good USB3 stick rather than the SD card
  — model loading dominates start-up time.
- Add `arm_boost=1` to `/boot/firmware/config.txt` and use active cooling, or
  the Pi will throttle mid-conversation.
- A USB microphone is far easier than a HAT. Check it appears in
  `companionai --list-devices`.
- Image generation is installed only with `--with-image`; a 384px render takes
  several minutes. It is genuinely usable for a one-off portrait, not for
  images during a conversation.
- If speech recognition is slow, drop to `tiny.en` and set the beam size to 1
  (the default).

## NVIDIA Jetson (Orin Nano / Orin NX / Xavier / Nano)

This is the board the real-time path was written for: the language model, the
Whisper model and the diffusion model all sit on the iGPU.

```bash
./scripts/install.sh
```

The installer creates the virtual environment with `--system-site-packages`, so
JetPack's CUDA-enabled `torch` is used instead of a CPU wheel from PyPI, and
builds `llama-cpp-python` with `-DGGML_CUDA=on`.

Notes:

- Install NVIDIA's torch wheel for your JetPack version **before** running the
  installer if you want image generation:
  <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/>
- `sudo nvpmodel -m 0 && sudo jetson_clocks` puts the board in its top power
  mode; without it latency roughly doubles.
- Leave **GPU layers** at `-1` (offload everything) in the Studio tab. If the
  model does not fit, step down a model size rather than reducing the layers —
  a partially offloaded model is slower than a smaller fully offloaded one.
- The original Jetson Nano (Maxwell, JetPack 4.6, Python 3.6) cannot run this
  app's Python requirements. Orin Nano and newer are the supported targets;
  the older board works only with a separately built Python 3.9+ toolchain.
- The iGPU shares system memory. If image generation ever gets killed, press
  *Free memory* on the System tab before rendering, or use `sd-turbo` at 512px.

## PC (Linux, Windows, macOS)

```bash
./scripts/install.sh          # Linux / macOS
```

On Windows use WSL2, or install by hand:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-voice.txt -r requirements-image.txt
pip install -e . --no-deps
python -m companionai
```

Notes:

- With an NVIDIA GPU the installer builds `llama-cpp-python` with CUDA. For a
  prebuilt wheel instead, see the llama-cpp-python README.
- On macOS, build with Metal:
  `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python`.
- Without a GPU everything still works; expect a couple of seconds before the
  companion starts speaking and about a minute for an image.

## Running it headless

Use the systemd unit in `scripts/companionai.service` and reach the UI from
another machine:

```bash
sudo cp scripts/companionai.service /etc/systemd/system/companionai@.service
sudo systemctl enable --now companionai@$USER
```

The unit binds `0.0.0.0`. That exposes the interface to your LAN — it has no
password, so only do it on a network you trust.

## Choosing models for a board

The Models tab marks every entry with whether it suits the detected hardware.
As a rule of thumb:

| board | language model | speech recognition | image |
| --- | --- | --- | --- |
| Pi 5 (4 GB) | 0.5B–1.5B Q4 | tiny.en | tiny-sd, or none |
| Pi 5 (8 GB) | 1.5B–3B Q4 | tiny.en / base.en | tiny-sd |
| Jetson Orin Nano | 3B Q4 | small.en | sd-turbo |
| PC, no GPU | 1.5B–3B Q4 | base.en | sd-turbo |
| PC with GPU | 3B+ Q4 | small.en / distil | sdxl-turbo |
