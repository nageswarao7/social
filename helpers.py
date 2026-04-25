"""utils/helpers.py — shared utilities for SocialWell AI Streamlit app."""

import asyncio
import uuid
from datetime import datetime


def run_async(coro):
    """Run an async coroutine safely from Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


def generate_session_id() -> str:
    return f"sw-{uuid.uuid4().hex[:10]}"


def format_timestamp() -> str:
    return datetime.now().strftime("%H:%M")


def detect_agent(message: str) -> tuple[str, str, str]:
    """
    Returns (agent_label, emoji, color) based on message content.
    Used for UI display before the actual agent responds.
    """
    msg = message.lower()

    if any(k in msg for k in ["viral", "trend", "strategy", "week", "plan", "pipeline", "what should i post"]):
        return "Content Pipeline", "🚀", "#f97316"
    elif any(k in msg for k in ["write", "caption", "post", "thread", "script", "create content", "draft", "generate"]):
        return "Content Creator", "✍️", "#38bdf8"
    elif any(k in msg for k in ["sentiment", "comment", "perception", "review", "audience feeling", "brand image"]):
        return "Sentiment Analyst", "📊", "#22c55e"
    elif any(k in msg for k in ["reply", "dm", "respond", "engage", "outreach", "collab", "community", "negative"]):
        return "Engagement Coach", "💬", "#a78bfa"
    elif any(k in msg for k in ["audience", "persona", "grow", "follower", "demographic", "target", "niche"]):
        return "Audience Insight", "👥", "#fb7185"
    elif any(k in msg for k in ["brand", "voice", "identity", "bio", "positioning", "pillar", "tone", "style"]):
        return "Brand Voice", "🎨", "#fbbf24"
    else:
        return "AI Coordinator", "🤖", "#38bdf8"


# ── Quick prompt examples ────────────────────────────────────────────────────
QUICK_PROMPTS = {
    "🚀 Viral Content Plan": (
        "Research trending topics in the personal development niche this week "
        "and create a full optimised content package for Instagram, LinkedIn, and Twitter."
    ),
    "✍️ Write Posts": (
        "Write engaging posts for all platforms about the topic: "
        "'Why most people never achieve their goals — and the simple fix'. "
        "Include hooks, hashtags, and CTAs."
    ),
    "📊 Analyse Sentiment": (
        "Analyse the current audience sentiment around AI tools for creators. "
        "What are people loving, hating, and asking for?"
    ),
    "💬 Reply Templates": (
        "Give me 10 engaging comment reply templates for a SaaS brand "
        "that handles both positive praise and negative complaints professionally."
    ),
    "👥 Audience Personas": (
        "Build detailed audience personas for a fitness coach targeting "
        "busy professionals aged 30-45 who want to lose weight without going to the gym."
    ),
    "🎨 Brand Voice": (
        "Help me define a strong brand voice for a solo founder building "
        "a B2B SaaS tool. I want to sound expert but approachable, not corporate."
    ),
}

# ── Platform colours ─────────────────────────────────────────────────────────
PLATFORM_COLORS = {
    "Twitter/X": "#1d9bf0",
    "LinkedIn": "#0a66c2",
    "Instagram": "#e1306c",
    "YouTube": "#ff0000",
    "TikTok": "#69c9d0",
    "Facebook": "#1877f2",
}

AGENT_META = {
    "Content Creator":   {"icon": "✍️", "color": "#38bdf8", "desc": "Posts, captions, threads, scripts"},
    "Content Pipeline":  {"icon": "🚀", "color": "#f97316", "desc": "Trend research → draft → optimise"},
    "Sentiment Analyst": {"icon": "📊", "color": "#22c55e", "desc": "Audience perception & brand health"},
    "Engagement Coach":  {"icon": "💬", "color": "#a78bfa", "desc": "Replies, DMs, community management"},
    "Audience Insight":  {"icon": "👥", "color": "#fb7185", "desc": "Personas, demographics, growth"},
    "Brand Voice":       {"icon": "🎨", "color": "#fbbf24", "desc": "Identity, tone, bio, positioning"},
}