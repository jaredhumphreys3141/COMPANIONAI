"""Look and feel.

Kept deliberately small: a soft built-in Gradio theme plus a handful of rules
so the interface stays readable on a 7-inch Raspberry Pi touchscreen and on a
desktop monitor, in light and dark mode alike.
"""

from __future__ import annotations

CSS = """
.companion-header { display:flex; align-items:center; gap:.75rem; flex-wrap:wrap; }
.companion-header h1 { margin:0; font-size:1.35rem; letter-spacing:-0.01em; }
.net-badge { padding:.2rem .6rem; border-radius:999px; font-size:.8rem; font-weight:600;
             border:1px solid currentColor; white-space:nowrap; }
.net-offline { color:#2e7d32; }
.net-online  { color:#c77700; }
.status-line { font-size:.85rem; opacity:.8; min-height:1.2em; }
.chat-window { min-height: 46vh; }
footer { display:none !important; }
@media (max-width: 820px) {
  .companion-header h1 { font-size:1.1rem; }
  .chat-window { min-height: 38vh; }
}
"""


def theme(name: str = "soft"):
    import gradio as gr  # noqa: PLC0415

    themes = {
        "soft": gr.themes.Soft,
        "default": gr.themes.Default,
        "monochrome": gr.themes.Monochrome,
        "glass": gr.themes.Glass,
    }
    builder = themes.get(name, gr.themes.Soft)
    return builder(primary_hue="teal", secondary_hue="slate")
