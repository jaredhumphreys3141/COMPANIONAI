"""Look and feel.

Kept deliberately small: a soft built-in Gradio theme plus a handful of rules
so the interface stays readable on a 7-inch Raspberry Pi touchscreen and on a
desktop monitor, in light and dark mode alike.
"""

from __future__ import annotations

CSS = """
.companion-header { display:flex; align-items:center; gap:.75rem; flex-wrap:wrap; }
/* Without this, flex children refuse to shrink below their content width and
   the network badge overlaps the status text on a narrow screen. */
.companion-header > * { min-width: 0 !important; flex: 0 1 auto; }
/* Gradio gives each row child a min-width, which pushes the badge onto its own
   line.  Let the title and badge size to their content and share one line. */
.companion-header > *:nth-child(-n+2) { flex: 0 0 auto; width: auto !important; }
.companion-header h1 { margin:0; font-size:1.35rem; letter-spacing:-0.01em; }
.net-badge { display:inline-block; padding:.2rem .6rem; border-radius:999px;
             font-size:.8rem; font-weight:600; border:1px solid currentColor;
             white-space:nowrap; }
.net-badge-short { display:none; }
.net-offline { color:#2e7d32; }
.net-online  { color:#c77700; }
.companion-subtle { font-size:.85rem; opacity:.75; }
.status-line { font-size:.85rem; opacity:.8; min-height:1.2em; }
.chat-window { min-height: 46vh; }
footer { display:none !important; }

/* Small screens - the 7" Raspberry Pi touchscreen is 800x480, so the header
   has to give up the long badge text and let the status line take its own
   row rather than collide with it. */
@media (max-width: 820px) {
  .companion-header { gap:.4rem; }
  .companion-header h1 { font-size:1.1rem; }
  .net-badge { font-size:.7rem; padding:.15rem .45rem; }
  .net-badge-long { display:none; }
  .net-badge-short { display:inline; }
  .companion-subtle { flex-basis:100%; margin:0; font-size:.75rem; }
  .chat-window { min-height: 38vh; }
}

/* Short screens: 480px tall leaves very little room once the header and the
   input row are drawn, so reclaim vertical padding. */
@media (max-height: 560px) {
  .chat-window { min-height: 30vh; }
  .companion-header { gap:.35rem; }
  .gradio-container { padding-top:.25rem !important; }
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
