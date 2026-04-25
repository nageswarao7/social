"""
SocialWell AI — agents/social_agents.py

Multi-agent architecture using Google ADK:

  SocialMediaCoordinator  (LlmAgent — root orchestrator)
       ├── ContentCreatorAgent        (LlmAgent — generates posts, captions, threads)
       ├── ContentPipeline            (SequentialAgent — 3-step content workflow)
       │       ├── TrendResearcher    (LlmAgent — finds trending topics)
       │       ├── ContentDrafter     (LlmAgent — drafts platform-specific content)
       │       └── ContentOptimizer   (LlmAgent — SEO, hashtags, CTA, scheduling)
       ├── SentimentAnalystAgent      (LlmAgent — analyses audience sentiment)
       ├── EngagementCoachAgent       (LlmAgent — reply suggestions, DM templates)
       ├── AudienceInsightAgent       (LlmAgent — persona, growth strategy)
       └── BrandVoiceAgent            (LlmAgent — brand consistency, tone guardian)
"""

import asyncio
import os
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"
APP_NAME = "socialwell_app"

# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST AGENTS
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Content Creator Agent ──────────────────────────────────────────────────
content_creator_agent = LlmAgent(
    name="ContentCreatorAgent",
    model=GEMINI_MODEL,
    description=(
        "Use when the user wants to create social media content: posts, captions, "
        "Twitter/X threads, LinkedIn articles, Instagram captions, YouTube descriptions, "
        "story scripts, or any platform-specific content creation WITHOUT trend research."
    ),
    instruction=(
        "You are an expert social media content creator with 10+ years of experience "
        "across all major platforms. Create compelling, platform-native content.\n\n"
        "For every request:\n"
        "  ✍️ **Platform Variants** — tailor content for each requested platform\n"
        "  📱 **Twitter/X** — punchy, max 280 chars, thread format if needed\n"
        "  💼 **LinkedIn** — professional, storytelling, thought leadership\n"
        "  📸 **Instagram** — visual-first caption, emojis, line breaks\n"
        "  🎬 **YouTube** — hook + description + timestamps if applicable\n"
        "  #️⃣ **Hashtags** — 5-10 relevant hashtags per platform\n"
        "  📣 **CTA** — clear call-to-action for each post\n"
        "  ⏰ **Best Posting Time** — recommend optimal time for each platform\n\n"
        "Match the user's brand voice and audience. Be creative, engaging, and authentic. "
        "Use Google Search to verify current platform best practices."
    ),
    tools=[google_search],
)

# ── 2. Content Pipeline (SequentialAgent) ────────────────────────────────────

trend_researcher = LlmAgent(
    name="TrendResearcher",
    model=GEMINI_MODEL,
    description="Pipeline step 1 — researches trending topics for the user's niche.",
    instruction=(
        "You are a social media trend analyst. Research what is currently trending "
        "for the user's niche/industry using Google Search.\n\n"
        "Deliver:\n"
        "  🔥 **Top 5 Trending Topics** — with search volume context\n"
        "  📈 **Rising Keywords** — terms gaining traction this week\n"
        "  🌊 **Viral Content Formats** — what content style is performing best\n"
        "  🤝 **Competitor Wins** — what top accounts are posting successfully\n"
        "  ⚡ **Content Gaps** — opportunities competitors are missing\n\n"
        "Always use Google Search for real-time trend data."
    ),
    output_key="trend_research",
    tools=[google_search],
)

content_drafter = LlmAgent(
    name="ContentDrafter",
    model=GEMINI_MODEL,
    description="Pipeline step 2 — drafts content aligned with trends.",
    instruction=(
        "You are a viral content writer. Based on the trend research below, "
        "draft high-quality social media content that rides the trends.\n\n"
        "Trend Research:\n{trend_research}\n\n"
        "Create:\n"
        "  📝 **3 Post Variations** — different angles on the trending topic\n"
        "  🧵 **Twitter Thread** — 5-7 tweet thread on the hottest trend\n"
        "  💼 **LinkedIn Post** — thought leadership piece on the trend\n"
        "  📸 **Instagram Caption** — engaging, visually described content\n"
        "  💡 **Content Hook Ideas** — 5 attention-grabbing opening lines\n\n"
        "Make content feel organic and authentic, not trend-chasing. "
        "Prioritise value for the audience."
    ),
    output_key="content_drafts",
    tools=[google_search],
)

content_optimizer = LlmAgent(
    name="ContentOptimizer",
    model=GEMINI_MODEL,
    description="Pipeline step 3 — optimises content for reach, SEO, and engagement.",
    instruction=(
        "You are a social media growth expert. Optimise the drafted content for "
        "maximum reach, engagement, and algorithmic performance.\n\n"
        "Trend Research: {trend_research}\n"
        "Content Drafts: {content_drafts}\n\n"
        "Deliver a complete optimised content package:\n\n"
        "## 🚀 Optimised Posts\n"
        "Refined versions of each draft with improved hooks and CTAs\n\n"
        "## #️⃣ Hashtag Strategy\n"
        "Mix of: 3 mega (1M+), 4 mid (100K-1M), 3 niche (<100K) hashtags per platform\n\n"
        "## ⏰ Publishing Schedule\n"
        "Best times and days for each platform this week\n\n"
        "## 📊 Predicted Performance\n"
        "Estimated reach, engagement rate, and growth impact\n\n"
        "## 🔄 Repurposing Plan\n"
        "How to turn this content into 5+ pieces across platforms\n\n"
        "## 🎯 A/B Test Suggestions\n"
        "2 variants to test for the best-performing post"
    ),
    tools=[google_search],
)

content_pipeline = SequentialAgent(
    name="ContentPipeline",
    description=(
        "Use when the user wants trend-driven content creation with full optimisation. "
        "Best for: 'create viral content about X', 'what should I post this week', "
        "'build me a content strategy', 'help me go viral in [niche]'."
    ),
    sub_agents=[trend_researcher, content_drafter, content_optimizer],
)

# ── 3. Sentiment Analyst Agent ────────────────────────────────────────────────
sentiment_analyst_agent = LlmAgent(
    name="SentimentAnalystAgent",
    model=GEMINI_MODEL,
    description=(
        "Use when the user wants to understand audience sentiment, analyse comments, "
        "understand how their brand/content is perceived, or analyse competitor sentiment."
    ),
    instruction=(
        "You are an expert social media sentiment analyst and brand perception specialist.\n\n"
        "Analyse and deliver:\n"
        "  📊 **Overall Sentiment Score** — Positive / Neutral / Negative breakdown (%)\n"
        "  💬 **Common Themes** — what people are saying (categories with examples)\n"
        "  ❤️ **What Resonates** — content and messaging that gets positive reactions\n"
        "  ⚠️ **Pain Points** — complaints, frustrations, unmet expectations\n"
        "  🌟 **Brand Advocates** — how to identify and engage superfans\n"
        "  🔥 **Crisis Signals** — early warning signs of negative momentum\n"
        "  📈 **Sentiment Trend** — is perception improving or declining?\n"
        "  🎯 **Actionable Fixes** — specific changes to improve sentiment\n\n"
        "Use Google Search to research public sentiment data. "
        "Be honest and objective — give real insights, not just positives."
    ),
    tools=[google_search],
)

# ── 4. Engagement Coach Agent ─────────────────────────────────────────────────
engagement_coach_agent = LlmAgent(
    name="EngagementCoachAgent",
    model=GEMINI_MODEL,
    description=(
        "Use when the user needs help responding to comments, writing DMs, handling "
        "negative feedback, increasing engagement, community management, or building "
        "relationships with followers and collaborators."
    ),
    instruction=(
        "You are a community management expert and engagement strategist.\n\n"
        "Help with:\n"
        "  💬 **Comment Replies** — craft authentic, engaging responses to comments\n"
        "  📩 **DM Templates** — collaboration outreach, partnership pitches, fan replies\n"
        "  😤 **Negative Feedback Handling** — de-escalate, respond professionally\n"
        "  🤝 **Influencer Outreach** — personalised pitch messages\n"
        "  🎯 **Engagement Boosters** — questions, polls, challenges to spark interaction\n"
        "  📣 **Community Building** — strategies to build loyal audience communities\n"
        "  🔄 **Re-engagement** — win back inactive followers\n"
        "  ⭐ **UGC Strategy** — encourage user-generated content\n\n"
        "Always sound human, warm, and authentic. "
        "Avoid corporate language. Match the user's brand voice."
    ),
    tools=[google_search],
)

# ── 5. Audience Insight Agent ─────────────────────────────────────────────────
audience_insight_agent = LlmAgent(
    name="AudienceInsightAgent",
    model=GEMINI_MODEL,
    description=(
        "Use when the user wants to understand their target audience, build audience "
        "personas, develop a growth strategy, analyse demographics, find their ideal "
        "followers, or plan platform-specific growth tactics."
    ),
    instruction=(
        "You are an audience research specialist and social media growth strategist.\n\n"
        "Deliver:\n"
        "  👥 **Audience Personas** — 2-3 detailed ICP (Ideal Content Person) profiles\n"
        "    Each includes: age, occupation, pain points, goals, content preferences,\n"
        "    active hours, platforms used, what makes them follow/unfollow\n"
        "  📊 **Platform Demographics** — which platforms your audience is on and why\n"
        "  🎯 **Content That Converts** — formats and topics that attract YOUR audience\n"
        "  📈 **Growth Roadmap** — 30/60/90 day follower growth strategy\n"
        "  🔍 **Where to Find Them** — hashtags, communities, subreddits, forums\n"
        "  🤝 **Collaboration Targets** — types of accounts to partner with\n"
        "  📉 **Churn Analysis** — why people unfollow and how to prevent it\n\n"
        "Use Google Search for platform demographic data and industry benchmarks."
    ),
    tools=[google_search],
)

# ── 6. Brand Voice Agent ──────────────────────────────────────────────────────
brand_voice_agent = LlmAgent(
    name="BrandVoiceAgent",
    model=GEMINI_MODEL,
    description=(
        "Use when the user wants to define or refine their brand voice, build a "
        "personal brand, create brand guidelines, ensure content consistency, "
        "develop a bio/profile, or craft their unique positioning."
    ),
    instruction=(
        "You are a personal branding expert and brand strategist for social media.\n\n"
        "Deliver:\n"
        "  🎨 **Brand Voice Definition** — 3-5 tone descriptors with examples of each\n"
        "  📖 **Brand Story** — compelling narrative for the bio and About pages\n"
        "  ✅ **Do's and Don'ts** — specific language and content guidelines\n"
        "  🎯 **Unique Value Proposition** — what makes them stand out in their niche\n"
        "  📝 **Bio Templates** — platform-optimised bios for Twitter, LinkedIn, Instagram\n"
        "  🎙️ **Content Pillars** — 4-5 recurring themes to build authority around\n"
        "  🌈 **Visual Direction** — colour, style, and aesthetic recommendations\n"
        "  📏 **Consistency Checklist** — how to audit content for brand alignment\n\n"
        "Make the brand feel authentic and human, not corporate. "
        "Use Google Search to research the user's niche and successful brand examples."
    ),
    tools=[google_search],
)

# ─────────────────────────────────────────────────────────────────────────────
# ROOT COORDINATOR
# ─────────────────────────────────────────────────────────────────────────────
coordinator = LlmAgent(
    name="SocialMediaCoordinator",
    model=GEMINI_MODEL,
    description="Root orchestrator for SocialWell AI — social media enhancement system.",
    instruction=(
        "You are SocialWell AI — an elite social media enhancement assistant. "
        "Analyse the user's message and route to the right specialist agent.\n\n"
        "Routing guide:\n"
        "  • ContentCreatorAgent    → quick post/caption creation, specific platform content\n"
        "  • ContentPipeline        → trend-driven content strategy, viral content, weekly plans\n"
        "  • SentimentAnalystAgent  → audience sentiment, brand perception, comment analysis\n"
        "  • EngagementCoachAgent   → replies, DMs, community management, outreach\n"
        "  • AudienceInsightAgent   → target audience, personas, growth strategy, demographics\n"
        "  • BrandVoiceAgent        → brand identity, voice, bio, positioning, content pillars\n\n"
        "If the query is ambiguous, ask ONE clarifying question before routing. "
        "Always be enthusiastic about helping creators grow their social media presence."
    ),
    sub_agents=[
        content_creator_agent,
        content_pipeline,
        sentiment_analyst_agent,
        engagement_coach_agent,
        audience_insight_agent,
        brand_voice_agent,
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
_session_store: dict = {}


def _get_or_create_session(session_id: str, user_id: str) -> dict:
    if session_id not in _session_store:
        service = InMemorySessionService()
        runner = Runner(agent=coordinator, app_name=APP_NAME, session_service=service)
        _session_store[session_id] = {"service": service, "runner": runner}

    return _session_store[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
async def run_social_coordinator(
    user_message: str,
    session_id: str = "default",
    user_id: str = "creator-1",
    conversation_history: list[dict] | None = None,
) -> str:
    """Run the SocialWell multi-agent system for a given user message."""

    if conversation_history and len(conversation_history) > 1:
        recent = conversation_history[-6:]
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent[:-1]
        )
        full_prompt = (
            f"[Recent conversation]\n{history_text}\n\n"
            f"[Current message]\nUser: {user_message}"
        )
    else:
        full_prompt = user_message

    store = _get_or_create_session(session_id, user_id)
    runner: Runner = store["runner"]
    service: InMemorySessionService = store["service"]

    try:
        await service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception:
        pass

    message = types.Content(role="user", parts=[types.Part(text=full_prompt)])

    final_response = None
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=message
        ):
            if event.content and event.content.parts:
                t = event.content.parts[0].text
                if t:
                    final_response = t
    except Exception as e:
        return f"⚠️ Agent error: {e}\n\nPlease check your API key and try again."

    return final_response or "No response received. Please try again."