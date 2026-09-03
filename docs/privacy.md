# What leaves this machine

Short answer: nothing, unless you press a button that says it will.

## The default

Every launch starts offline. `net.init()` runs before the interface is built,
resets any "allow for this session" grant from last time, and sets
`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE` so the
libraries underneath refuse network calls too.

## The one exception

Downloading a model. That is the only code path in the app that opens a socket
to the outside world, and it goes through `companionai/net.py`:

1. `require()` refuses unless you approved access on the Settings tab.
2. The host is checked against the allow list (`huggingface.co` and friends by
   default; edit it, or empty it to allow any host).
3. The attempt is appended to `~/.companionai/logs/network-audit.jsonl`, with
   the result, whether it succeeded or was blocked.
4. Downloads land in a `.part` file and are checksummed where the catalogue
   gives a checksum, so a broken transfer never masquerades as a model.

Approving access lets you *pull* models. It does not send anything: no
conversation, no audio, no image, no usage statistics, no crash report.

## Approval scopes

| scope | meaning |
| --- | --- |
| Offline | everything blocked (the default, and the state after every restart) |
| Allow for this session | permitted until you close the app |
| Allow and remember | permitted across restarts, until you set it back |

## Third-party telemetry

`companionai/__init__.py` sets these before Gradio, HuggingFace or Torch are
imported: `GRADIO_ANALYTICS_ENABLED=False`, `HF_HUB_DISABLE_TELEMETRY=1`,
`DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`.

## Where your data sits

| what | where |
| --- | --- |
| companions | `~/.companionai/characters/*.json` |
| conversations | `~/.companionai/transcripts/*.json` (turn it off in Settings → Runtime) |
| images and their prompts | `~/.companionai/gallery/` |
| models | `~/.companionai/models/` |
| network activity | `~/.companionai/logs/network-audit.jsonl` |

Delete the directory and nothing remains. Move it with `COMPANIONAI_HOME` or
`--home`.

## Things that are not private

- **`--host 0.0.0.0`** puts the interface on your LAN. Set a password first —
  Settings → Runtime → Sign-in. Without one, anyone who can reach the port can
  talk to your companion and read its transcripts, and the app prints a warning
  saying so at start-up. Only the salted hash of the password is stored
  (PBKDF2-SHA256), never the password itself.
- **`--share`** opens a public tunnel through Gradio's servers. It is off by
  default and prints a warning when you use it. The models still run locally,
  but the web interface is then reachable from the internet.
- **Models you download** come from whoever published them and carry their own
  licences, listed in the Models tab.

## Verifying the pipeline

```bash
companionai --smoke-test --allow-network
```

Installs the three smallest models, generates a reply, speaks it with Piper,
transcribes that audio back with Whisper, and checks the words survived the
round trip. It approves network access for the download only, then revokes it
again — you can watch both in the activity log.

## Checking for yourself

```bash
companionai --doctor          # prints the current policy
tail -f ~/.companionai/logs/network-audit.jsonl
```

Or watch at the OS level while you use the app — with the policy set to
Offline you should see no outbound connections at all:

```bash
sudo ss -tunp | grep python
```
