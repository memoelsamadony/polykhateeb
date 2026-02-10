"""
TV-optimized viewer for real-time transcription display.

Hub page: shows links to per-language viewers.
Per-language pages: teleprompter-style animated text for one language.
Raw Arabic page: real-time ASR output.

Access via:
  Hub:        http://<laptop-ip>:8501/?mode=tv
  English:    http://<laptop-ip>:8501/?mode=tv&lang=en
  German:     http://<laptop-ip>:8501/?mode=tv&lang=de
  Arabic:     http://<laptop-ip>:8501/?mode=tv&lang=ar
  Raw Arabic: http://<laptop-ip>:8501/?mode=tv&lang=raw
"""

import time

import streamlit as st
import streamlit.components.v1 as components

from .shared_state import get_shared_state

# How many recent blocks to show (teleprompter window)
TELEPROMPTER_WINDOW = 5

# How often each page refreshes (seconds)
TV_POLL_INTERVAL = 0.5

# Language configuration: key -> (label, snapshot_key, css_direction)
LANG_CONFIG = {
    "en": ("English", "refined_en", "ltr"),
    "de": ("Deutsch", "refined_de", "ltr"),
    "ar": ("العربية", "refined_ar", "rtl"),
    "raw": ("العربية — خام", "raw_ar", "rtl"),
}


# ---------------------------------------------------------------------------
# Hub page
# ---------------------------------------------------------------------------

def _hub_css():
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri&family=Inter:wght@400;600;700&display=swap');

.stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], header, .main {
    background-color: #0a0a0a !important;
    color: #e0e0e0 !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="collapsedControl"],
header[data-testid="stHeader"],
footer,
#MainMenu,
.stDeployButton {
    display: none !important;
}

.block-container {
    padding: 2rem 1rem !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

.hub-title {
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0f0f0;
    margin-bottom: 0.3rem;
}
.hub-subtitle {
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #666;
    margin-bottom: 2.5rem;
}

.hub-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
    max-width: 700px;
    margin: 0 auto;
}

.hub-card {
    display: block;
    text-decoration: none;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 2rem 1.5rem;
    transition: all 0.25s ease;
    text-align: center;
}
.hub-card:hover {
    background: #1e1e1e;
    border-color: #444;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}

.hub-card .card-flag {
    font-size: 2.8rem;
    margin-bottom: 0.6rem;
}
.hub-card .card-label {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f0f0f0;
    margin-bottom: 0.3rem;
}
.hub-card .card-label.rtl {
    font-family: 'Amiri', 'Traditional Arabic', serif;
    font-size: 1.6rem;
}
.hub-card .card-desc {
    font-size: 0.85rem;
    color: #888;
}

.hub-card.raw-card {
    grid-column: 1 / -1;
    background: #1a1208;
    border-color: #3d2e0a;
}
.hub-card.raw-card:hover {
    background: #241a0c;
    border-color: #5a4310;
}

.hub-status {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #555;
}
.hub-status .live-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: #ff3333;
    border-radius: 50%;
    margin-right: 6px;
    animation: livePulse 1.5s ease-in-out infinite;
}
@keyframes livePulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.3; }
}
</style>
"""


def _run_hub():
    """Render the hub page with links to each language viewer."""
    shared = get_shared_state()
    snapshot = shared.get_snapshot(last_n=1)

    st.markdown(_hub_css(), unsafe_allow_html=True)

    # Build base URL from current page
    # Streamlit query params make this straightforward: just link with ?mode=tv&lang=xx
    html = """
<div class="hub-title">Khutbah Live</div>
<div class="hub-subtitle">Select a language to view the live transcription</div>

<div class="hub-grid">
    <a class="hub-card" href="?mode=tv&lang=en" target="_blank">
        <div class="card-flag">🇬🇧</div>
        <div class="card-label">English</div>
        <div class="card-desc">Refined translation</div>
    </a>
    <a class="hub-card" href="?mode=tv&lang=de" target="_blank">
        <div class="card-flag">🇩🇪</div>
        <div class="card-label">Deutsch</div>
        <div class="card-desc">Refined translation</div>
    </a>
    <a class="hub-card" href="?mode=tv&lang=ar" target="_blank">
        <div class="card-flag">🇸🇦</div>
        <div class="card-label rtl">العربية</div>
        <div class="card-desc">Refined Arabic text</div>
    </a>
    <a class="hub-card raw-card" href="?mode=tv&lang=raw" target="_blank">
        <div class="card-flag">🎙️</div>
        <div class="card-label rtl">العربية — بث مباشر خام</div>
        <div class="card-desc">Raw real-time ASR output (unprocessed)</div>
    </a>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)

    # Status
    if snapshot["is_streaming"]:
        st.markdown(
            '<div class="hub-status">'
            '<span class="live-dot"></span> LIVE'
            f' &mdash; Session {snapshot["session_uid"]}'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="hub-status">Waiting for sermon to begin...</div>',
            unsafe_allow_html=True,
        )

    # Auto-refresh to update status
    time.sleep(3)
    st.rerun()


# ---------------------------------------------------------------------------
# Per-language viewer
# ---------------------------------------------------------------------------

def _lang_css(direction: str):
    """CSS for a single-language teleprompter view."""
    is_rtl = direction == "rtl"
    font_family = "'Amiri', 'Traditional Arabic', 'Noto Sans Arabic', serif" if is_rtl else "'Segoe UI', 'Roboto', 'Inter', sans-serif"
    text_align = "right" if is_rtl else "left"
    dir_attr = "rtl" if is_rtl else "ltr"
    font_size = "2.8rem" if is_rtl else "2.4rem"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri&family=Inter:wght@400;600&display=swap');

.stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], header, .main {{
    background-color: #0a0a0a !important;
    color: #e0e0e0 !important;
}}

[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="collapsedControl"],
header[data-testid="stHeader"],
footer,
#MainMenu,
.stDeployButton {{
    display: none !important;
}}

.block-container {{
    padding: 0.5rem 1.5rem !important;
    max-width: 100% !important;
}}

.lang-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #222;
    margin-bottom: 0.5rem;
}}
.lang-header .lang-title {{
    font-family: {font_family};
    font-size: 1.1rem;
    font-weight: 700;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}}
.lang-header .back-link {{
    font-size: 0.85rem;
    color: #555;
    text-decoration: none;
}}
.lang-header .back-link:hover {{
    color: #aaa;
}}

.tv-block {{
    font-family: {font_family};
    font-size: {font_size};
    line-height: 1.6;
    direction: {dir_attr};
    text-align: {text_align};
    padding: 0.6rem 1rem;
    color: #f0f0f0;
    border-bottom: 1px solid #1a1a1a;
}}

.tv-block.new {{
    animation: tvFadeIn 0.7s ease-out;
}}

.tv-block.age-0 {{ opacity: 1.0; }}
.tv-block.age-1 {{ opacity: 0.75; }}
.tv-block.age-2 {{ opacity: 0.50; }}
.tv-block.age-3 {{ opacity: 0.30; }}
.tv-block.age-4 {{ opacity: 0.18; }}

@keyframes tvFadeIn {{
    from {{ opacity: 0; transform: translateY(18px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.tv-status {{
    text-align: center;
    font-size: 0.85rem;
    color: #666;
    padding: 0.3rem;
}}
.tv-status .live-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    background: #ff3333;
    border-radius: 50%;
    margin-right: 6px;
    animation: livePulse 1.5s ease-in-out infinite;
}}
@keyframes livePulse {{
    0%, 100% {{ opacity: 1; }}
    50%      {{ opacity: 0.3; }}
}}

.tv-waiting {{
    text-align: center;
    font-size: 2.2rem;
    color: #555;
    padding-top: 25vh;
}}

.stMarkdown {{ margin-bottom: 0 !important; }}
</style>
"""


def _inject_auto_scroll():
    """Scroll the page to the bottom so newest content is visible."""
    components.html(
        """
        <script>
        const main = window.parent.document.querySelector('.main');
        if (main) main.scrollTop = main.scrollHeight;
        </script>
        """,
        height=0,
    )


def _run_lang_viewer(lang: str):
    """Render a single-language teleprompter view."""
    label, snapshot_key, direction = LANG_CONFIG[lang]

    shared = get_shared_state()
    snapshot = shared.get_snapshot(last_n=TELEPROMPTER_WINDOW)

    # Track previous count for animation targeting
    state_key = f"tv_prev_{lang}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    prev_count = st.session_state[state_key]

    blocks = snapshot[snapshot_key]

    st.markdown(_lang_css(direction), unsafe_allow_html=True)

    # Header with back link
    st.markdown(
        f'<div class="lang-header">'
        f'<a class="back-link" href="?mode=tv">&larr; All languages</a>'
        f'<span class="lang-title">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Status bar
    if snapshot["is_streaming"]:
        st.markdown(
            '<div class="tv-status">'
            '<span class="live-dot"></span> LIVE'
            f' &mdash; Session {snapshot["session_uid"]}'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="tv-waiting">Waiting for sermon to begin...</div>',
            unsafe_allow_html=True,
        )

    # Render blocks with animation
    if snapshot["is_streaming"] and blocks:
        n = len(blocks)
        html_parts = []
        for i, block in enumerate(blocks):
            age = (n - 1) - i
            age_class = f"age-{min(age, 4)}"
            new_class = "new" if (n - i) > prev_count else ""
            html_parts.append(
                f'<div class="tv-block {age_class} {new_class}">{block}</div>'
            )
        st.markdown("\n".join(html_parts), unsafe_allow_html=True)
        _inject_auto_scroll()

    # Save current count
    st.session_state[state_key] = len(blocks)

    # Auto-refresh
    time.sleep(TV_POLL_INTERVAL)
    st.rerun()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_tv_viewer():
    """Main entry point for TV viewer mode.

    Routes based on the ``lang`` query parameter:
    - (none) -> hub page with links
    - en/de/ar/raw -> per-language teleprompter view
    """
    lang = st.query_params.get("lang", "")

    if lang in LANG_CONFIG:
        _run_lang_viewer(lang)
    else:
        _run_hub()
