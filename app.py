"""
SocialWell AI — app.py
Agentic AI for enhancing social media user experiences.
Powered by Google ADK + Gemini 2.0 Flash.

Run: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from helpers import (
    run_async,
    generate_session_id,
    format_timestamp,
    detect_agent,
    QUICK_PROMPTS,
    AGENT_META,
    PLATFORM_COLORS,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SocialWell AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES — clean, modern, light aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #ffffff; color: #1e293b; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

.main .block-container {
    padding: 1.5rem 2.5rem 3rem;
    max-width: 920px;
}

/* Hero */
.hero {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #f97316;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    line-height: 1.05;
    color: #0f172a;
    letter-spacing: -0.03em;
    margin: 0 0 1rem;
}
.hero-title em { font-style: normal; color: #f97316; }
.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto 1.5rem;
    line-height: 1.6;
}
.hero-pills {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
}
.hero-pill {
    font-size: 0.72rem;
    padding: 5px 14px;
    border-radius: 100px;
    border: 1px solid #e2e8f0;
    color: #475569;
    background: #f1f5f9;
    letter-spacing: 0.04em;
}

/* Stats */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.75rem;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f97316;
}
.stat-lbl {
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
}

/* Agent cards in sidebar */
.ag-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 11px 14px;
    margin-bottom: 8px;
    cursor: default;
}
.ag-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 3px;
}
.ag-name { font-size: 0.82rem; font-weight: 600; }
.ag-desc { font-size: 0.72rem; color: #64748b; line-height: 1.4; }

/* Messages */
.msg-wrap { margin-bottom: 1.75rem; }
.msg-meta { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.msg-avatar {
    width: 34px; height: 34px;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.msg-avatar.usr { background: #fff7ed; border: 1px solid #fed7aa; }
.msg-avatar.bot { background: #f0f9ff; border: 1px solid #bae6fd; }
.msg-name { font-size: 0.8rem; font-weight: 600; color: #475569; }
.msg-time { font-size: 0.7rem; color: #cbd5e1; }
.bubble {
    border-radius: 4px 16px 16px 16px;
    padding: 15px 20px;
    font-size: 0.91rem;
    line-height: 1.75;
    color: #334155;
    border: 1px solid #e2e8f0;
    background: #ffffff;
}
.bubble.usr {
    background: #fff7ed;
    border-color: #fed7aa;
    border-radius: 16px 4px 16px 16px;
}

/* Quick prompts */
.qp-label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500; margin-bottom: 12px; }

/* Platform pills */
.plt-pill { padding: 5px 14px; border-radius: 100px; font-size: 0.75rem; font-weight: 500; border: 1px solid; }

/* Input overrides */
.stTextArea textarea { border: 1px solid #e2e8f0 !important; border-radius: 14px !important; }

/* Buttons */
.stButton > button { border: 1px solid #e2e8f0; color: #475569; border-radius: 10px; }
.stButton > button[kind="primary"] { background: #f97316; border-color: #f97316; color: #fff; }

/* Divider */
hr { border-color: #e2e8f0 !important; }

/* Markdown in bubbles */
.bubble h1,.bubble h2,.bubble h3 { color: #1a2540; margin:1rem 0 .5rem; font-family:'Syne',sans-serif; }
.bubble h2 { font-size: 1.05rem; }
.bubble strong { color: #0f172a; }
.bubble code { background: #f1f5f9; padding: 2px 6px; border-radius:4px; font-size:.83rem; color:#c2410c; }
.bubble ul,.bubble ol { padding-left:1.25rem; }
.bubble li { margin-bottom: 4px; }

/* API warning */
.api-warn {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.83rem;
    color: #c2410c;
    margin-bottom: 1rem;
}

/* Session tag */
.sess-tag {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.72rem;
    color: #94a3b8;
    margin-bottom: .75rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f5f7fa; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

/* Spinner */
.stSpinner > div { border-top-color: #f97316 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "session_id": generate_session_id(),
    "messages": [],
    "handle": "Creator",
    "total_queries": 0,
    "agents_used": set(),
    "api_key_set": bool(os.getenv("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    "platforms": ["Twitter/X", "LinkedIn", "Instagram"],
    "niche": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem">
        <div style="font-size:2.2rem;margin-bottom:6px">🚀</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:800;color:#0f172a;letter-spacing:-0.02em">
            Social<span style="color:#f97316">Well</span> AI
        </div>
        <div style="font-size:0.68rem;color:#64748b;letter-spacing:.1em;text-transform:uppercase;margin-top:4px">
            Powered by Google ADK
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── API Key ──
    if not st.session_state.api_key_set:
        st.markdown("**🔑 API Key**")
        api_key = st.text_input("Google API Key", type="password", placeholder="AIza...",
                                help="Get free key at aistudio.google.com")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.session_state.api_key_set = True
            st.success("✅ Connected!")
            st.rerun()
        st.markdown(
            "<div style='font-size:.72rem;color:#3d5573;margin-top:5px'>"
            "Free key → <a href='https://aistudio.google.com' style='color:#f97316'>aistudio.google.com</a></div>",
            unsafe_allow_html=True
        )
        st.divider()

    # ── Creator Profile ──
    st.markdown("**👤 Your Profile**")
    handle = st.text_input("Your name / handle", value=st.session_state.handle, placeholder="@yourhandle")
    if handle != st.session_state.handle:
        st.session_state.handle = handle

    niche = st.text_input("Your niche", value=st.session_state.niche,
                          placeholder="e.g. SaaS, Fitness, Finance…")
    if niche != st.session_state.niche:
        st.session_state.niche = niche

    st.markdown("**📱 Active Platforms**")
    all_platforms = list(PLATFORM_COLORS.keys())
    selected = st.multiselect(
        "Platforms",
        all_platforms,
        default=st.session_state.platforms,
        label_visibility="collapsed",
    )
    if selected != st.session_state.platforms:
        st.session_state.platforms = selected

    st.divider()

    # ── Session Stats ──
    st.markdown("**📊 Session**")
    c1, c2 = st.columns(2)
    c1.metric("Messages", st.session_state.total_queries)
    c2.metric("Agents", len(st.session_state.agents_used))

    st.markdown(
        f"<div class='sess-tag'>ID: <code style='color:#f97316'>{st.session_state.session_id[-8:]}</code></div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Agent Directory ──
    st.markdown("**🤖 AI Agent Directory**")
    for name, meta in AGENT_META.items():
        st.markdown(f"""
        <div class="ag-card">
            <div class="ag-head">
                <span style="font-size:.95rem">{meta['icon']}</span>
                <span class="ag-name" style="color:{meta['color']}">{name}</span>
            </div>
            <div class="ag-desc">{meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = generate_session_id()
        st.session_state.total_queries = 0
        st.session_state.agents_used = set()
        st.rerun()

    st.markdown("""
    <div style="font-size:.68rem;color:#475569;margin-top:.75rem;padding:10px;
                background:#ffffff;border-radius:8px;border:1px solid #e2e8f0;line-height:1.5">
        🤖 <strong style="color:#334155">6 specialist agents</strong> orchestrated by Google ADK.
        Content is AI-generated — review before publishing.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# ── Hero (empty state) ────────────────────────────────────────────────────────
if not st.session_state.messages:
    platforms_str = " · ".join(st.session_state.platforms) if st.session_state.platforms else "All platforms"

    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Agentic AI for Social Media</div>
        <h1 class="hero-title">Grow Smarter,<br>Post <em>Better</em></h1>
        <p class="hero-subtitle">
            6 specialist AI agents working together to create viral content,
            analyse your audience, and supercharge your social media presence.
        </p>
        <div class="hero-pills">
            <span class="hero-pill">✍️ Content Creation</span>
            <span class="hero-pill">🚀 Trend Pipeline</span>
            <span class="hero-pill">📊 Sentiment Analysis</span>
            <span class="hero-pill">💬 Engagement</span>
            <span class="hero-pill">👥 Audience Insights</span>
            <span class="hero-pill">🎨 Brand Voice</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div class="stats-row">
        <div class="stat-card"><div class="stat-num">6</div><div class="stat-lbl">AI Agents</div></div>
        <div class="stat-card"><div class="stat-num">3</div><div class="stat-lbl">Pipeline Steps</div></div>
        <div class="stat-card"><div class="stat-num">6</div><div class="stat-lbl">Platforms</div></div>
        <div class="stat-card"><div class="stat-num">∞</div><div class="stat-lbl">Content Ideas</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Quick prompts
    st.markdown("<div class='qp-label'>✦ Try one of these</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (label, prompt) in enumerate(QUICK_PROMPTS.items()):
        with cols[i % 3]:
            if st.button(label, key=f"qp_{i}", use_container_width=True):
                st.session_state.prefill = prompt
                st.rerun()

else:
    # Compact top bar when chatting
    platforms_html = "".join(
        f'<span class="plt-pill" style="color:{PLATFORM_COLORS.get(p,"#6b8aad")};'
        f'border-color:{PLATFORM_COLORS.get(p,"#6b8aad")}33;'
        f'background:{"rgba(" + ",".join(str(int(PLATFORM_COLORS.get(p,"#6b8aad").lstrip("#")[i:i+2],16)) for i in (0,2,4)) + ",0.06)"}">{ p}</span>'
        for p in st.session_state.platforms
    )
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;padding:.5rem 0 1.25rem;flex-wrap:wrap;gap:10px">
        <div style="display:flex;align-items:center;gap:10px">
            <span style="font-size:1.4rem">🚀</span>
            <span style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;color:#1a2540">
                Social<span style="color:#f97316">Well</span> AI
            </span>
        </div>
        <div class="platform-row" style="margin:0">{platforms_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ── API Key warning ───────────────────────────────────────────────────────────
if not st.session_state.api_key_set:
    st.markdown("""
    <div class="api-warn">
        🔑 <strong>Enter your Google API key in the sidebar</strong> to activate all 6 AI agents.
        Get a free key at <strong>aistudio.google.com</strong>.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    is_user = msg["role"] == "user"

    if is_user:
        st.markdown(f"""
        <div class="msg-wrap">
            <div class="msg-meta">
                <div class="msg-avatar usr">👤</div>
                <span class="msg-name">{st.session_state.handle}</span>
                <span class="msg-time">{msg.get('time','')}</span>
            </div>
            <div class="bubble usr">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        agent_name = msg.get("agent", "AI Coordinator")
        agent_icon = AGENT_META.get(agent_name, {}).get("icon", "🤖")
        agent_color = AGENT_META.get(agent_name, {}).get("color", "#38bdf8")

        # Convert color hex to rgb for rgba()
        hex_c = agent_color.lstrip("#")
        rgb = ",".join(str(int(hex_c[i:i+2], 16)) for i in (0, 2, 4))

        st.markdown(f"""
        <div class="msg-wrap">
            <div class="msg-meta">
                <div class="msg-avatar bot">{agent_icon}</div>
                <span class="msg-name">SocialWell AI</span>
                <span class="agent-chip" style="background:rgba({rgb},0.1);color:{agent_color};border:1px solid rgba({rgb},0.25)">{agent_name}</span>
                <span class="msg-time">{msg.get('time','')}</span>
            </div>
            <div class="bubble">
        """, unsafe_allow_html=True)

        st.markdown(msg["content"])

        st.markdown("</div></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.divider()

prefill = st.session_state.pop("prefill", "")

# Inject niche/platforms context into prompt
context_hint = ""
if st.session_state.niche:
    context_hint = f"[My niche: {st.session_state.niche}] "
if st.session_state.platforms:
    context_hint += f"[Platforms: {', '.join(st.session_state.platforms)}] "

with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([6, 1])

    with col_in:
        raw_input = st.text_area(
            "Message",
            value=prefill,
            placeholder="Ask anything — create content, analyse audience, build brand voice, craft replies…",
            height=88,
            label_visibility="collapsed",
        )

    with col_btn:
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Send →", type="primary", use_container_width=True)

# Hint buttons below input
hint_cols = st.columns(6)
for i, (label, prompt) in enumerate(QUICK_PROMPTS.items()):
    with hint_cols[i]:
        if st.button(label.split()[0] + " " + label.split()[1], key=f"h_{i}"):
            st.session_state.prefill = prompt
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
if submitted and raw_input.strip():
    if not st.session_state.api_key_set:
        st.error("Please add your Google API key in the sidebar.")
        st.stop()

    user_text = raw_input.strip()
    # Prepend context if niche/platforms are set
    enriched = (context_hint + user_text).strip() if context_hint else user_text
    timestamp = format_timestamp()

    agent_name, agent_icon, agent_color = detect_agent(user_text)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "time": timestamp,
    })
    st.session_state.total_queries += 1
    st.session_state.agents_used.add(agent_name)

    # Run agents
    with st.spinner(f"{agent_icon} {agent_name} is working…"):
        try:
            from agents import run_social_coordinator

            response = run_async(
                run_social_coordinator(
                    user_message=enriched,
                    session_id=st.session_state.session_id,
                    user_id=st.session_state.handle or "creator-1",
                    conversation_history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                )
            )

        except ImportError:
            response = f"""**⚙️ Demo Mode** — install dependencies to activate live agents.

```bash
pip install google-adk google-genai python-dotenv
streamlit run app.py
```

**Your request:** *{user_text}*

**Agent that would handle this:** {agent_icon} **{agent_name}**

Once connected, this agent will generate a full, detailed response tailored to your platforms ({', '.join(st.session_state.platforms)}) and niche ({st.session_state.niche or 'not set'}).

---
*Configure your Google API key and restart to use live AI agents.*"""

        except Exception as e:
            response = f"⚠️ **Error:** {e}\n\nCheck your API key and try again."

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "agent": agent_name,
        "time": format_timestamp(),
    })

    st.rerun()