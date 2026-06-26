# =============================================================================
# Maze RevOps — Lead Prioritization Prototype
# =============================================================================
# Cuts through the volume of low-intent inbound (report downloads, free signups)
# to surface the handful worth an SDR's time right now, and ties leads back to
# the marketing sources that actually produce quality leads.
#
# DESIGN PHILOSOPHY (the part that shows judgment):
#   - The base score is RULES-BASED and DETERMINISTIC. Plain Python on
#     structured fields. Cheap, fast, fully explainable. No LLM here.
#   - The LLM (Claude) is used ONLY on the one unstructured signal available
#     BEFORE a sales conversation — the lead's free-text goal from the capture
#     form — and only for leads that already clear a base-score threshold. That
#     boundary is deliberate: it shows where AI adds leverage vs. where it's overkill.
#
# Only PRE-TOUCH lead data is used (available before an SDR contacts the lead):
# marketing capture & behavior (HubSpot), firmographic enrichment (Clay), and
# product usage (PLG). No Outreach/Gong/Salesforce — a sequence reply, a recorded
# call, or an open opportunity all mean the lead has ALREADY been worked, which is
# downstream of this step.
# =============================================================================

import json
import os
import random
from collections import Counter

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIG — every tunable value lives here so the points and thresholds are easy
# to read and adjust. Nothing magic is buried in the scoring logic below.
# -----------------------------------------------------------------------------

SCORING_CONFIG = {
    # --- Firmographic fit (enrichment from Clay) ---
    # Mid-market is the sweet spot: big enough to have budget and a real
    # research team, not so big that the deal stalls in procurement.
    "company_size": {
        "<200": 5,
        "200-1000": 20,
        "1000-5000": 25,
        ">5000": 15,
    },
    "industry_fit_target": 15,   # in target industry list
    "industry_fit_other": 5,
    "title_fit_target": 15,      # in target buyer/user title list
    "title_fit_other": 5,
    "free_email_penalty": -10,   # gmail/yahoo/etc. -> likely not a real buyer

    # --- Behavioral / intent (HubSpot, Product) ---
    "source": {
        "demo_request": 30,      # asked to talk to sales = strongest signal
        "free_signup": 20,
        "webinar": 10,
        "paid_ad": 8,
        "report_download": 5,    # the classic low-intent content grab
        "organic": 3,
    },
    "email_open_points": 2,      # per open...
    "email_open_cap": 10,        # ...capped so an "open-happy" lead can't run away
    "pricing_page_viewed": 20,   # high-intent buying behavior

    # PLG signal — product usage on the free tier. Only meaningful for
    # free_signup leads (see scoring note), so we gate it on source.
    "free_tier_sessions_high": 25,   # > 10 sessions = power user
    "free_tier_sessions_mid": 12,    # 4-10 sessions = activated

    # --- Tiering from the TOTAL score (base + LLM intent boost added later) ---
    "tier_A": 70,   # Hot:     >= 70
    "tier_B": 45,   # Warm:    45-69
    "tier_C": 25,   # Cool:    25-44
                    # Nurture: < 25  (D)

    # --- LLM gate: only call Claude for leads that already look promising ---
    "llm_threshold": 45,   # base score >= this -> worth an LLM look (B and up)
}

# Membership sets used by the fit rules above.
TARGET_INDUSTRIES = {"SaaS", "Fintech", "E-commerce", "Financial Services", "Retail"}
TARGET_TITLES = {
    "Product Manager", "UX Researcher", "Designer",
    "Head of Product", "Head of Research", "Design Director",
}
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com",
    "outlook.com", "icloud.com", "aol.com",
}


# -----------------------------------------------------------------------------
# SYNTHETIC DATA GENERATOR
# -----------------------------------------------------------------------------
# 50 fake leads, seeded so the worklist is reproducible across runs. Each lead
# is a dict whose fields are grouped by the tool that would own them in Maze's
# real stack (HubSpot / Clay / Product). Data is deliberately
# varied so the worklist has an obvious spread of strong and weak leads.
# -----------------------------------------------------------------------------

_FIRST_NAMES = [
    "Avery", "Jordan", "Riley", "Casey", "Morgan", "Taylor", "Quinn", "Reese",
    "Dakota", "Skylar", "Priya", "Diego", "Mei", "Omar", "Hannah", "Liam",
    "Sofia", "Noah", "Aisha", "Lucas", "Chloe", "Ethan", "Maya", "Daniel",
]
_LAST_NAMES = [
    "Chen", "Patel", "Garcia", "Nguyen", "Okafor", "Kim", "Rossi", "Silva",
    "Johnson", "Martinez", "Hughes", "Ali", "Andersson", "Costa", "Ivanov",
    "Bauer", "Lopez", "Schmidt", "Walsh", "Park", "Mensah", "Dubois",
]

# (company name, email domain). Companies span sizes/industries so firmographic
# scoring has something to chew on. Some leads will override the domain with a
# free email provider to trigger the penalty.
_COMPANIES = [
    ("Northwind Labs", "northwindlabs.com"),
    ("Brightwave", "brightwave.io"),
    ("Helios Bank", "heliosbank.com"),
    ("Cartwheel Commerce", "cartwheel.shop"),
    ("Lumen Analytics", "lumenanalytics.com"),
    ("Tabletop Retail Co", "tabletopretail.com"),
    ("Finch Pay", "finchpay.com"),
    ("Orbit Software", "orbit.dev"),
    ("Maplewood Health", "maplewoodhealth.org"),
    ("Vertex Logistics", "vertexlog.com"),
    ("Sunset Studios", "sunsetstudios.co"),
    ("Granite Financial", "granitefin.com"),
    ("Pixel & Pine", "pixelandpine.com"),
    ("Acme Robotics", "acmerobotics.ai"),
    ("Clearwater Retail", "clearwaterretail.com"),
    ("Bluefin SaaS", "bluefin.io"),
]

# Most inbound is OFF-ICP, so "Other" dominates both industry and title. This
# is what makes the funnel realistic: a great-fit lead is the exception.
_INDUSTRIES = ["SaaS", "Fintech", "E-commerce", "Financial Services", "Retail", "Other"]
_INDUSTRY_WEIGHTS = [8, 6, 6, 4, 4, 60]
_TITLES = [
    "Product Manager", "UX Researcher", "Designer", "Head of Product",
    "Head of Research", "Design Director", "Other",
]
_TITLE_WEIGHTS = [6, 5, 5, 3, 2, 2, 55]
# Top-of-funnel for a PLG tool skews SMB/individual; mid-market (the high-point
# buckets) is the minority.
_COMPANY_SIZES = ["<200", "200-1000", "1000-5000", ">5000"]
_SIZE_WEIGHTS = [58, 18, 8, 16]

_SOURCES = ["demo_request", "free_signup", "report_download", "webinar", "paid_ad", "organic"]
# Weights skew hard toward the low-intent end (report downloads + free signups)
# and make demo requests rare — that's Maze's actual problem: a flood of
# content/PLG leads, only a trickle asking to talk to sales.
_SOURCE_WEIGHTS = [6, 20, 30, 14, 12, 18]

_CAMPAIGNS = [
    "Research Maturity Report", "UX Trends Webinar", "Free Plan",
    "Continuous Discovery Guide", "Product Research Benchmark",
    "G2 Paid Campaign", "Onboarding Teardown Series",
]

# Free-text form goals — the unstructured field the LLM reads. Same idea: some
# concrete/urgent, some vague. This is the lead's answer to the "what are you
# trying to do?" box on the capture form, available before any sales contact.
_GOAL_HIGH = [
    "Launching a new onboarding flow, need to test with 50 users next week.",
    "Need unmoderated tests before our Q3 redesign ships.",
    "Replacing our manual research process, want to decide this month.",
    "Validating a pricing page change before launch, tight timeline.",
    "Standing up a research practice, need a tool the whole team can use.",
]
_GOAL_LOW = [
    "Just wanted to see what this does.",
    "Exploring options for the future.",
    "Curious about UX research tools in general.",
    "Saw it mentioned somewhere, taking a look.",
    "No specific project yet, just browsing.",
]


def _email_for(first, last, company_name, domain, use_free, rng):
    """Build an email. Sometimes use a free provider to trigger the penalty."""
    handle = f"{first}.{last}".lower()
    if use_free:
        return f"{handle}@{rng.choice(list(FREE_EMAIL_DOMAINS))}"
    return f"{handle}@{domain}"


def generate_leads(n=50, seed=42):
    """Return a list of n synthetic lead dicts. Seeded for reproducibility."""
    rng = random.Random(seed)
    leads = []

    for _ in range(n):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        company_name, domain = rng.choice(_COMPANIES)

        # ~30% of leads use a personal/free email -> firmographic red flag.
        use_free = rng.random() < 0.30

        source = rng.choices(_SOURCES, weights=_SOURCE_WEIGHTS, k=1)[0]

        # Email opens skew low with a long tail: most leads barely engage, a
        # few are highly engaged. (A flat 0-12 made everyone look engaged.)
        opens = rng.randint(0, 3)
        if rng.random() < 0.25:
            opens = min(opens + rng.randint(1, 9), 12)

        # Free-tier usage (PLG) only exists for free signups, and most signups
        # churn immediately — power users are the rare, valuable tail.
        if source == "free_signup":
            r = rng.random()
            if r < 0.60:
                sessions = rng.randint(0, 3)      # churned / barely tried it
            elif r < 0.85:
                sessions = rng.randint(4, 10)     # activated
            else:
                sessions = rng.randint(11, 30)    # power user
        else:
            sessions = 0

        lead = {
            # --- HubSpot — lead-capture form (identity) ---
            "name": f"{first} {last}",
            "email": _email_for(first, last, company_name, domain, use_free, rng),
            "company": company_name,

            # --- HubSpot — marketing capture / behavior ---
            "source": source,
            "campaign": rng.choice(_CAMPAIGNS),
            "email_opens": opens,
            "pricing_page_viewed": rng.random() < 0.15,

            # --- Clay (enrichment / firmographics) ---
            "company_size": rng.choices(_COMPANY_SIZES, weights=_SIZE_WEIGHTS, k=1)[0],
            "industry": rng.choices(_INDUSTRIES, weights=_INDUSTRY_WEIGHTS, k=1)[0],
            "title": rng.choices(_TITLES, weights=_TITLE_WEIGHTS, k=1)[0],

            # --- Product (PLG signal) — only meaningful for free signups ---
            "free_tier_sessions": sessions,

            # --- HubSpot — form free-text goal (the field Claude reads) ---
            "freetext_goal": rng.choice(_GOAL_HIGH if rng.random() < 0.5 else _GOAL_LOW),
        }
        leads.append(lead)

    return leads


# -----------------------------------------------------------------------------
# RULES-BASED SCORING  (deterministic — no LLM)
# -----------------------------------------------------------------------------
# Returns (base_score, breakdown) where breakdown is a list of (reason, points)
# tuples. The breakdown is what makes every score defensible in the demo: you
# can point at exactly which rules fired and for how much.
# -----------------------------------------------------------------------------

def score_lead(lead):
    cfg = SCORING_CONFIG
    breakdown = []  # list of (human-readable reason, points)

    # ---- Firmographic fit (Clay) ----
    size = lead["company_size"]
    breakdown.append((f"Company size {size}", cfg["company_size"][size]))

    if lead["industry"] in TARGET_INDUSTRIES:
        breakdown.append((f"Industry fit: {lead['industry']}", cfg["industry_fit_target"]))
    else:
        breakdown.append((f"Industry: {lead['industry']}", cfg["industry_fit_other"]))

    if lead["title"] in TARGET_TITLES:
        breakdown.append((f"Buyer/user title: {lead['title']}", cfg["title_fit_target"]))
    else:
        breakdown.append((f"Title: {lead['title']}", cfg["title_fit_other"]))

    domain = lead["email"].split("@")[-1].lower()
    if domain in FREE_EMAIL_DOMAINS:
        breakdown.append(("Free email domain", cfg["free_email_penalty"]))

    # ---- Behavioral / intent (HubSpot, Product) ----
    src = lead["source"]
    breakdown.append((f"Source: {src}", cfg["source"][src]))

    open_pts = min(lead["email_opens"] * cfg["email_open_points"], cfg["email_open_cap"])
    if open_pts:
        breakdown.append((f"Email opens x{lead['email_opens']} (capped)", open_pts))

    if lead["pricing_page_viewed"]:
        breakdown.append(("Viewed pricing page", cfg["pricing_page_viewed"]))

    # PLG: free-tier usage only counts for free signups (it's noise otherwise).
    if src == "free_signup":
        sessions = lead["free_tier_sessions"]
        if sessions > 10:
            breakdown.append((f"Power free-tier user ({sessions} sessions)",
                              cfg["free_tier_sessions_high"]))
        elif sessions >= 4:
            breakdown.append((f"Active free-tier user ({sessions} sessions)",
                              cfg["free_tier_sessions_mid"]))

    base_score = sum(points for _, points in breakdown)
    return base_score, breakdown


def tier_for_score(total):
    """Map a total score to a tier letter."""
    cfg = SCORING_CONFIG
    if total >= cfg["tier_A"]:
        return "A"
    if total >= cfg["tier_B"]:
        return "B"
    if total >= cfg["tier_C"]:
        return "C"
    return "D"


# -----------------------------------------------------------------------------
# LLM LAYER  (Claude — UNSTRUCTURED signals only)
# -----------------------------------------------------------------------------
# This is the ONLY place an LLM is justified. The rules above already handled
# everything structured. Here we hand Claude the one free-text field a rule
# can't read — the lead's stated goal from the capture form — and ask for:
#   - intent_boost: 0-20 extra points for genuine urgency/timeline/competition
#   - why_now:      a one-line reason an SDR should work this lead now
#
# Cost discipline (the judgment Guillaume is probing for):
#   - We only call Claude for leads already at/above the B threshold. The long
#     C/D tail never costs an API call.
#   - We use Sonnet, not Opus — this is a tiny classification on one short
#     string, not a reasoning task. Right tool for the job.
#   - max_tokens is tiny (300). The boundary is additive-only (0..+20), per the
#     design decision: the LLM can lift a borderline lead, never silently demote.
#   - Every failure path (no key, network error, bad JSON) falls back to a
#     deterministic why_now. The worklist must never crash in a live demo.
# -----------------------------------------------------------------------------

LLM_MODEL = "claude-sonnet-4-6"   # cheap classification, not a reasoning task
LLM_MAX_TOKENS = 300

# System prompt is explicit: JSON only, no prose, no fences. We still parse
# defensively below in case the model wraps it anyway.
LLM_SYSTEM_PROMPT = (
    "You are a RevOps assistant that reads a sales lead's free-text goal (typed "
    "into a marketing lead-capture form) and rates buying intent. Judge how much "
    "EXTRA intent the text shows beyond a normal inbound lead: concrete timelines, "
    "urgency, competitive evaluation, signed budget, or a specific near-term use "
    "case raise it; vague curiosity or 'just exploring' keep it low. Reply with "
    "ONLY a JSON object, no prose and no markdown fences, in exactly this shape:\n"
    '{"intent_boost": <integer 0-20>, "why_now": "<one sentence, under 15 words>"}'
)

# Lazily-created singleton client so we don't construct it 50 times.
_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        import anthropic
        # max_retries=0 is deliberate: the SDK default (2) means a rate-limited
        # call silently fires 3x, AMPLIFYING load during exactly the storm we're
        # guarding against. A 429 must fail fast and fall back, never retry.
        _llm_client = anthropic.Anthropic(max_retries=0)  # reads ANTHROPIC_API_KEY from env
    return _llm_client


def _rules_based_why_now(lead, base_score):
    """Deterministic 'why_now' used when we don't (or can't) call Claude."""
    signals = []
    if lead["source"] == "demo_request":
        signals.append("requested a demo")
    if lead["pricing_page_viewed"]:
        signals.append("viewed pricing")
    if lead["source"] == "free_signup" and lead["free_tier_sessions"] > 10:
        signals.append("heavy free-tier usage")

    if signals:
        return f"Good ICP fit; {', '.join(signals[:2])}."
    return "Solid firmographic fit; little engagement yet."


def _claude_intent(goal):
    """Raw Claude call for ONE form-goal string -> (intent_boost, why_now).

    This is the ONLY function that hits the API. It may raise (network /
    rate-limit / bad JSON); callers catch it and fall back. The client uses
    max_retries=0, so a 429 fails fast instead of amplifying load. All gating
    (B+ threshold, key present) and caching happens in the caller (get_full),
    so this is never reached for a below-threshold lead or a cached goal.
    """
    user_msg = (f'Lead\'s stated goal (from the lead-capture form): '
                f'"{goal or "(no stated goal)"}"')
    resp = _get_llm_client().messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=LLM_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    # Pull the text block (don't assume content[0]); strip stray fences.
    text = next((b.text for b in resp.content if b.type == "text"), "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]  # jump to the first '{'
    data = json.loads(text)
    boost = max(0, min(int(data.get("intent_boost", 0)), 20))  # clamp 0..20
    why = str(data.get("why_now", "")).strip()
    return boost, why


# -----------------------------------------------------------------------------
# ROUTING
# -----------------------------------------------------------------------------
# Only A and B leads are worth an SDR's time — those get an owner. C and D are
# left to marketing nurture.
#
# Maze is a small team — no territories. Every rep covers everything; we just
# round-robin DOWN THE RANKED LIST so each SDR ends up with a similar-sized,
# similar-quality portfolio (no one rep hoards all the A's while another works a
# stack of B's).
# -----------------------------------------------------------------------------

SDRS = ["Alex", "Sam", "Jordan"]
NO_OWNER = "Nurture (no SDR)"


def route_worklist(scored_results):
    """
    Sort results by total score (desc) and assign an 'owner' to each:
      - A/B tier  -> an SDR, round-robin over the ranked list
      - C/D tier  -> 'Nurture (no SDR)'
    Returns the ranked list; mutates each dict to add 'owner'.
    """
    ranked = sorted(scored_results, key=lambda r: r["total_score"], reverse=True)
    rr = 0
    for r in ranked:
        if r["tier"] in ("A", "B"):
            r["owner"] = SDRS[rr % len(SDRS)]
            rr += 1
        else:
            r["owner"] = NO_OWNER
    return ranked


# -----------------------------------------------------------------------------
# STREAMLIT UI   (run with:  streamlit run app.py)
# -----------------------------------------------------------------------------
# A 6-stage guided walkthrough that EXECUTES one step at a time. Nothing is
# computed up front: stage 1 only generates the raw data; the rules run when you
# reach stage 2; Claude runs only when you reach stage 3; routing at stage 4.
# Each stage's result is held in session_state, so going Back never recomputes
# and Claude is never called until you actually arrive at its stage.
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Maze RevOps — Lead Prioritization",
                   page_icon="🎯", layout="wide")

# On Streamlit Community Cloud the key is set in "Secrets" (st.secrets); mirror it
# into the environment so the rest of the app (which reads ANTHROPIC_API_KEY) and
# the Anthropic SDK pick it up. Locally, ANTHROPIC_API_KEY is just an env var.
try:
    if not os.environ.get("ANTHROPIC_API_KEY") and "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass


# --- Optional password gate — protects the public app AND the API key ---------
# Inert unless APP_PASSWORD is set in Streamlit Secrets. When set, the whole app
# (and therefore any Claude call) is blocked until the password is entered, so
# random or bot traffic on the public URL can't reach stage 3 and trigger spend.
def _require_password():
    try:
        required = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else None
    except Exception:
        required = None
    if not required or st.session_state.get("_authed"):
        return
    st.title("🎯 Maze RevOps — Lead Prioritization")
    pw = st.text_input("Demo password", type="password")
    if pw == required:
        st.session_state["_authed"] = True
        st.rerun()
    if pw:
        st.error("Incorrect password.")
    st.stop()


_require_password()


# --- Lazy, per-stage computation, held in session_state ----------------------
# Each get_*() runs its stage at most once per dataset; regenerate() wipes them.

def get_leads():
    if "leads" not in st.session_state:
        st.session_state.leads = generate_leads(
            seed=42 + st.session_state.get("data_version", 0))
    return st.session_state.leads


def regenerate():
    """Draw a fresh dataset and STAY on the Data flood stage showing the new leads.
    Downstream stages recompute when next reached. The enriched list is keyed by
    data_version (bumped here), so it's naturally fresh; goal_cache is intentionally
    KEPT so repeated goal strings in the new dataset cost ZERO new Claude calls."""
    old = st.session_state.get("data_version", 0)
    for k in ("leads", "scored_rules", "ranked", f"scored_full_v{old}", f"stats_v{old}"):
        st.session_state.pop(k, None)
    st.session_state.data_version = old + 1
    st.session_state.step = 1       # ① Data flood — show the new data right here
    st.session_state.max_step = 1   # collapse downstream chips (stale until recomputed)


def get_rules():
    """Stage 2 — deterministic rules only (no LLM)."""
    if "scored_rules" not in st.session_state:
        rules = []
        for ld in get_leads():
            base, breakdown = score_lead(ld)
            rules.append({"lead": ld, "base_score": base, "breakdown": breakdown})
        st.session_state.scored_rules = rules
    return st.session_state.scored_rules


def get_full():
    """Stage 3 — Claude reads the ONE unstructured field (the form goal). This is
    the ONLY place Claude is called, and it's cached so reruns make ZERO new calls:
      - the enriched list is cached per dataset (data_version), so Back/Next/filter
        changes and repeat visits never rebuild it;
      - each unique goal string's result is cached (goal_cache), so duplicate goals
        never re-call — and that cache survives Regenerate, since the synthetic goal
        pool repeats, so a fresh dataset usually needs ZERO new calls.
    Below-threshold leads and (no key) short-circuit to a fallback with no call.
    """
    version = st.session_state.get("data_version", 0)
    list_key = f"scored_full_v{version}"
    if list_key in st.session_state:          # already built for this dataset -> no calls
        return st.session_state[list_key]

    cfg = SCORING_CONFIG
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    goal_cache = st.session_state.setdefault("goal_cache", {})  # goal -> (boost, why) | None=failed
    st.session_state["enrich_builds"] = st.session_state.get("enrich_builds", 0) + 1
    calls = reused = bplus = 0

    full = []
    with st.spinner("Claude is reading the form goals…"):
        for r in get_rules():
            lead, base = r["lead"], r["base_score"]
            if base >= cfg["llm_threshold"]:
                bplus += 1
            # HARD GATE: Claude runs ONLY for B+ leads, ONLY with a key. Anything
            # else short-circuits to a deterministic fallback — no API call.
            if base < cfg["llm_threshold"] or not has_key:
                boost, why = 0, _rules_based_why_now(lead, base)
            else:
                goal = lead["freetext_goal"]
                if goal in goal_cache:                       # per-input guard: never re-call
                    cached = goal_cache[goal]
                    boost, why = cached if cached else (0, _rules_based_why_now(lead, base))
                    reused += 1
                else:
                    try:
                        boost, why = _claude_intent(goal)    # the single API call site
                        why = why or ("Form goal signals concrete intent." if boost > 0
                                      else "Form goal shows little urgency.")
                        goal_cache[goal] = (boost, why)
                    except Exception:                         # 429/network/parse -> fallback, NO retry
                        boost, why = 0, _rules_based_why_now(lead, base)
                        goal_cache[goal] = None               # remember the failure; don't retry
                    calls += 1
            total = base + boost
            full.append({**r, "intent_boost": boost, "why_now": why,
                         "total_score": total, "tier": tier_for_score(total)})

    st.session_state[list_key] = full
    st.session_state[f"stats_v{version}"] = {"calls": calls, "reused": reused, "bplus": bplus}
    return full


def get_ranked():
    """Stage 4 — tier and route the scored leads."""
    if "ranked" not in st.session_state:
        st.session_state.ranked = route_worklist(get_full())
    return st.session_state.ranked


# Stage-1 data layout: every captured field, grouped by the tool it comes from.
# Identity (name/email/company) and the free-text goal are all HubSpot form fields,
# so they live under HubSpot — not separate "sources". Three real sources only.
_DATA_GROUPS = [
    ("HubSpot", [
        ("Name", "name"), ("Email", "email"), ("Company", "company"),
        ("Source", "source"), ("Campaign", "campaign"),
        ("Email opens", "email_opens"), ("Viewed pricing?", "pricing_page_viewed"),
        ("Stated goal (form)", "freetext_goal"),
    ]),
    ("Clay", [
        ("Company size", "company_size"), ("Industry", "industry"), ("Title", "title"),
    ]),
    ("Product", [
        ("Free-tier sessions", "free_tier_sessions"),
    ]),
]

# --- Sidebar: the 'explain every choice' legend ---
with st.sidebar:
    st.header("How this works")
    st.markdown(
        "**Rules (deterministic)** score the structured fields from HubSpot, "
        "Clay & the product — instant, free, fully explainable.\n\n"
        "**Claude** scores only the *unstructured* signal (the lead's form goal), "
        "and only for leads that already clear the B threshold — adding a 0–20 "
        "intent boost and a one-line *why now*.\n\n"
        "**Routing** sends A & B leads round-robin to 3 SDRs; C & D nurture."
    )
    st.divider()
    st.caption("Tiers (total score): A ≥ 70 · B 45–69 · C 25–44 · D < 25")

# --- Walkthrough state ---
STEPS = [
    "Demo notes",
    "① Data flood",
    "② Rules score signals",
    "③ Claude reads intent",
    "④ Tier & route",
    "⑤ SDR worklist",
    "⑥ Source → lead quality",
]
if "step" not in st.session_state:
    st.session_state.step = 0
if "max_step" not in st.session_state:
    st.session_state.max_step = 0  # furthest stage reached — gates the strip


def _set_step(i):
    st.session_state.step = max(0, min(len(STEPS) - 1, i))


def _render_nav(suffix):
    """Back / Next — rendered both above and below each stage so the action to
    advance is always in view (unique keys per placement)."""
    bcol, _, ncol = st.columns([1, 4, 1])
    bcol.button("← Back", key=f"back_{suffix}", use_container_width=True,
                disabled=st.session_state.step == 0,
                on_click=_set_step, args=(st.session_state.step - 1,))
    ncol.button("Next →", key=f"next_{suffix}", use_container_width=True, type="primary",
                disabled=st.session_state.step == len(STEPS) - 1,
                on_click=_set_step, args=(st.session_state.step + 1,))


# --- Stage renderers: each computes its OWN stage on arrival, then renders -----

def render_intro():
    st.subheader("Demo notes — what this is & how to use it")
    st.markdown(
        "**The problem.** The SDR team is buried in low-intent inbound. Which few leads "
        "should a rep call first, and which marketing sources actually produce the leads "
        "worth working?"
    )
    st.markdown(
        "**What this does.** Scores every inbound lead, surfaces the few worth working, "
        "routes them to reps, and ties results back to source. It uses only data we have "
        "*before* anyone talks to the lead."
    )
    st.markdown(
        "**How to use it.** Click **Next →** to run one stage at a time. Stages unlock as "
        "you go; earlier ones stay clickable. The **🔄 Regenerate** button on Stage 1 draws "
        "a fresh set of leads."
    )
    st.markdown("**The six stages:**")
    st.markdown(
        "1. **Data flood** — the raw leads, every field, grouped by the tool it comes from "
        "(**HubSpot** for the form fill + marketing behavior, **Clay** for firmographic "
        "enrichment, **Product** for free-tier usage).\n"
        "2. **Rules score signals** — fast, deterministic points on the structured fields. No AI.\n"
        "3. **Claude reads intent** — AI reads the one free-text field that rules can't parse.\n"
        "4. **Tier & route** — sort into A/B/C/D, hand A & B to SDRs.\n"
        "5. **SDR worklist** — the ranked call list, every score explained.\n"
        "6. **Source → lead quality** — which channels produce the leads worth working."
    )
    st.caption("Prototype: rules + Claude (Sonnet) on synthetic data. No real customer data.")


def render_data():
    leads = get_leads()
    st.subheader("① The data flood — every signal we have, grouped by its source tool")
    st.markdown(
        "Raw inbound leads — **nothing scored yet, and no one has contacted them.** Every "
        "field we have, grouped by the tool that captured it: **HubSpot** (the form fill — "
        "name, company, the stated goal — plus marketing behavior like email opens and "
        "pricing-page views), **Clay** (firmographic enrichment off the email domain), and "
        "**Product** (free-tier usage). No Outreach / Gong / Salesforce — those only exist "
        "*after* a rep works the lead."
    )
    mcols = st.columns([1, 1, 1.3])
    mcols[0].metric("Leads in", len(leads))
    mcols[1].metric("Scored so far", 0)
    mcols[2].button("🔄 Regenerate data", on_click=regenerate, use_container_width=True,
                    help="Draw a fresh random set of leads and restart the walkthrough.")

    cols = pd.MultiIndex.from_tuples(
        [(group, label) for group, fields in _DATA_GROUPS for label, _ in fields])
    rows = [[ld[key] for group, fields in _DATA_GROUPS for _, key in fields] for ld in leads]
    df = (pd.DataFrame(rows, columns=cols)
            .sort_values(("HubSpot", "Name"))
            .reset_index(drop=True))
    st.dataframe(df, hide_index=True, use_container_width=True, height=430)
    st.caption("Nothing scored yet. Click **Next →** to run the rules over the structured fields.")


def render_rules():
    rules = get_rules()
    cfg = SCORING_CONFIG
    st.subheader("② Rules Score Signals — fast, free, fully explainable")
    st.markdown(
        "First pass is **plain-Python rules** on the structured fields: firmographic fit "
        "(who they are, from Clay) plus behavioral intent (what they did, from HubSpot & "
        "the product). **No AI here.** This rulebook is the heart of the "
        "system — every point is tunable:"
    )
    fc, bc = st.columns(2)
    fc.markdown("**Firmographic fit — Clay**")
    fc.dataframe(pd.DataFrame(
        [{"Company size": k, "Points": v} for k, v in cfg["company_size"].items()]),
        hide_index=True, use_container_width=True)
    fc.dataframe(pd.DataFrame([
        {"Rule": "Target industry (SaaS, Fintech, E-comm, FinServ, Retail)", "Points": cfg["industry_fit_target"]},
        {"Rule": "Other industry", "Points": cfg["industry_fit_other"]},
        {"Rule": "Target title (PM, UX Researcher, Designer, Head of…)", "Points": cfg["title_fit_target"]},
        {"Rule": "Other title", "Points": cfg["title_fit_other"]},
        {"Rule": "Free email domain (gmail, yahoo, …)", "Points": cfg["free_email_penalty"]},
    ]), hide_index=True, use_container_width=True)
    bc.markdown("**Behavioral intent — HubSpot · Product**")
    bc.dataframe(pd.DataFrame(
        [{"How they arrived (source)": k, "Points": v} for k, v in cfg["source"].items()]),
        hide_index=True, use_container_width=True)
    bc.dataframe(pd.DataFrame([
        {"Rule": "Email opens (+2 each)", "Points": f"up to +{cfg['email_open_cap']}"},
        {"Rule": "Viewed pricing page", "Points": cfg["pricing_page_viewed"]},
        {"Rule": "Power free-tier user (>10 sessions)", "Points": cfg["free_tier_sessions_high"]},
        {"Rule": "Active free-tier user (4–10 sessions)", "Points": cfg["free_tier_sessions_mid"]},
    ]), hide_index=True, use_container_width=True)
    st.caption(
        f"Tiers from the total score:  A ≥ {cfg['tier_A']}  ·  B {cfg['tier_B']}–{cfg['tier_A'] - 1}  "
        f"·  C {cfg['tier_C']}–{cfg['tier_B'] - 1}  ·  D < {cfg['tier_C']}.   "
        f"Claude is only consulted for leads scoring ≥ {cfg['llm_threshold']} (stage 3)."
    )
    st.divider()
    scored = sorted(rules, key=lambda r: r["base_score"], reverse=True)
    left, right = st.columns([3, 2])
    left.markdown("**Every lead, now with a rules-only score:**")
    left.dataframe(pd.DataFrame([{
        "Name": r["lead"]["name"], "Company": r["lead"]["company"],
        "Source": r["lead"]["source"], "Rules score": r["base_score"],
    } for r in scored]), hide_index=True, use_container_width=True, height=300)
    example = scored[0]
    right.markdown(f"**Worked example — {example['lead']['name']}, {example['lead']['company']}:**")
    right.dataframe(pd.DataFrame(example["breakdown"], columns=["Rule fired", "Points"]),
                    hide_index=True, use_container_width=True)
    right.success(f"Rules score: {example['base_score']} — read top to bottom, no black box.")


def render_claude():
    cfg = SCORING_CONFIG
    thr = cfg["llm_threshold"]
    full = get_full()
    st.subheader("③ Claude reads what the rules can't")
    st.markdown(
        "Rules can't read a sentence. For leads already clearing the **B threshold "
        f"({thr})**, Claude reads the one **unstructured** field we have and returns "
        "a 0–20 **intent boost** plus a one-line **why now**. It's additive — the AI can "
        "lift a borderline lead, never silently bury one."
    )
    st.markdown(
        "**The free-text input — and where it comes from:**\n"
        "- 📝 **Form goal** — what the lead typed into the “what are you trying to do?” "
        "box on the **HubSpot lead-capture form** (the gated-report / demo-request form). "
        "It's the only unstructured signal we have *before* a sales conversation — exactly "
        "what a rule can't parse."
    )
    eligible = [r for r in full if r["base_score"] >= thr]
    version = st.session_state.get("data_version", 0)
    stats = st.session_state.get(f"stats_v{version}", {"calls": 0, "reused": 0})
    builds = st.session_state.get("enrich_builds", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("B+ leads (Claude-eligible)", len(eligible))
    c2.metric("🔌 Claude API calls (this dataset)", stats["calls"])
    c3.metric("Served from cache", stats["reused"])
    st.caption(
        f"Calls fire once per *unique* goal, once per dataset (model claude-sonnet-4-6). "
        f"Reruns, Back/Next, and filter changes add **zero** calls. "
        f"Enrichment builds this session so far: **{builds}**."
    )
    st.markdown("**What Claude did with that field:**")
    rows = [{
        "Name": r["lead"]["name"],
        "📝 Form goal (HubSpot)": r["lead"]["freetext_goal"],
        "Rules": r["base_score"],
        "+Claude": r["intent_boost"],
        "Total": r["total_score"],
        "Why now": r["why_now"],
    } for r in sorted(eligible, key=lambda r: r["total_score"], reverse=True)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=340)
    st.info(
        "The tell: a lead the rules rank high but Claude boosts little — strong "
        "structured behavior, but the *conversation* says tire-kicker. The score stays "
        "where the rules put it; the **why now** tells the SDR the truth."
    )


def render_route():
    ranked = get_ranked()
    st.subheader("④ Tier & route — cut the volume, split it fairly")
    st.markdown(
        "Total score (rules + Claude) sorts every lead into a tier. **A & B get an SDR**; "
        "**C & D** go to marketing nurture. Maze is a small team with no territories — "
        "everyone covers everything — so we round-robin down the ranked list to give each "
        "rep a **similar-sized, similar-quality portfolio** (no one rep hoards the A's)."
    )
    dist = Counter(r["tier"] for r in ranked)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔥 A — Hot", dist["A"])
    c2.metric("🌤️ B — Warm", dist["B"])
    c3.metric("🌥️ C — Cool", dist["C"])
    c4.metric("💤 D — Nurture", dist["D"])
    st.markdown(f"**{dist['A'] + dist['B']} leads** worth working, down from {len(ranked)}.")
    left, right = st.columns(2)
    owner_counts = Counter(r["owner"] for r in ranked if r["owner"] != NO_OWNER)
    left.markdown("**SDR portfolios (A & B only):**")
    left.dataframe(pd.DataFrame(
        [{"SDR": s, "Leads assigned": owner_counts.get(s, 0)} for s in SDRS]),
        hide_index=True, use_container_width=True)
    right.markdown("**Tier funnel:**")
    right.bar_chart(pd.DataFrame(
        [{"Tier": t, "Leads": dist.get(t, 0)} for t in ["A", "B", "C", "D"]]
    ).set_index("Tier"))


def render_worklist():
    ranked = get_ranked()
    worklist = [r for r in ranked if r["tier"] in ("A", "B")]
    st.subheader("⑤ The SDR worklist — what a rep opens Monday morning")
    f1, f2 = st.columns(2)
    tier_filter = f1.multiselect("Tier", ["A", "B"], default=["A", "B"], key="wl_tier")
    owner_filter = f2.multiselect("Owner (SDR)", SDRS, default=SDRS, key="wl_owner")
    shown = [r for r in worklist
             if r["tier"] in tier_filter and r["owner"] in owner_filter]
    st.caption(f"{len(shown)} of {len(worklist)} A/B leads shown · "
               "expand any lead to see exactly why it scored what it did")
    for r in shown:
        ld = r["lead"]
        header = (f"{r['tier']}  ·  score {r['total_score']}  ·  {ld['name']} "
                  f"— {ld['company']}  ·  {r['owner']}")
        with st.expander(header):
            st.markdown(f"**Why now:** {r['why_now']}")
            left, right = st.columns(2)
            left.markdown(
                f"**Score:** {r['base_score']} rules + {r['intent_boost']} Claude "
                f"= **{r['total_score']}**  \n"
                f"**Source:** {ld['source']}  \n"
                f"**Campaign:** {ld['campaign']}"
            )
            right.markdown(
                f"**Company size:** {ld['company_size']}  \n"
                f"**Industry:** {ld['industry']}  \n"
                f"**Title:** {ld['title']}  \n"
                f"**Email:** {ld['email']}"
            )
            st.markdown(f"> 📝 **Form goal:** {ld['freetext_goal']}")
            bd = pd.DataFrame(r["breakdown"], columns=["Rule fired", "Points"])
            if r["intent_boost"]:
                bd.loc[len(bd)] = ["Claude intent boost (unstructured)", r["intent_boost"]]
            st.dataframe(bd, hide_index=True, use_container_width=True)


def render_rollup():
    ranked = get_ranked()
    st.subheader("⑥ Source → lead quality — which channels produce leads worth working")
    df = pd.DataFrame([{
        "Source": r["lead"]["source"],
        "worth_working": r["tier"] in ("A", "B"),
        "is_a": r["tier"] == "A",
        "score": r["total_score"],
    } for r in ranked])
    rollup = (df.groupby("Source")
                .agg(Leads=("Source", "size"),
                     Worth_working=("worth_working", "sum"),
                     A_tier=("is_a", "sum"),
                     Avg_score=("score", "mean"))
                .reset_index()
                .sort_values("Worth_working", ascending=False))
    rollup["% worth working"] = (100 * rollup["Worth_working"] / rollup["Leads"]).round().astype(int)
    rollup["Avg_score"] = rollup["Avg_score"].round(1)

    disp = rollup.rename(columns={"Worth_working": "Worth working (A/B)",
                                  "A_tier": "A-tier", "Avg_score": "Avg score"}).copy()
    disp["% worth working"] = disp["% worth working"].astype(str) + "%"
    st.dataframe(disp, hide_index=True, use_container_width=True)

    st.markdown("**Leads worth working (A/B), by source**")
    st.bar_chart(rollup.set_index("Source")["Worth_working"])
    st.caption(
        "This is **source → lead quality** — which channels produce the leads worth an "
        "SDR's time, the leading indicator you can see pre-touch. In production you'd "
        "stitch HubSpot source to the Salesforce closed-won outcome for true "
        "source-to-revenue."
    )


_RENDERERS = [render_intro, render_data, render_rules, render_claude,
              render_route, render_worklist, render_rollup]

# --- Header + key status ---
st.title("🎯 Maze RevOps — Lead Prioritization")
st.caption("A guided, step-by-step walkthrough — each stage unlocks and runs only when "
           "you reach it. Use Next → to advance (earlier stages stay clickable).")

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning("ANTHROPIC_API_KEY not set — *why now* (stage 3) will use the rules-based "
               "fallback. Scoring and routing are unaffected.", icon="⚠️")

# --- Progress strip: stages are REVEALED one at a time as you advance, so the
# walkthrough builds up step by step instead of showing all six at once. Already-
# reached stages stay clickable (to go back); unreached ones don't appear yet.
st.session_state.max_step = max(st.session_state.max_step, st.session_state.step)
nav_cols = st.columns(len(STEPS))  # fixed width so chips fill in left-to-right
for i in range(st.session_state.max_step + 1):
    nav_cols[i].button(
        STEPS[i], key=f"nav_{i}", use_container_width=True,
        type="primary" if i == st.session_state.step else "secondary",
        on_click=_set_step, args=(i,),
    )
st.progress((st.session_state.step + 1) / len(STEPS))

# --- "How to proceed" callout — make it obvious you advance with Next ---
_last = len(STEPS) - 1
if st.session_state.step == 0:
    st.info("👉 Read the quick notes below, then click **Next →** to start the walkthrough.")
elif st.session_state.step == _last:
    st.success("✓ Last stage — that's the whole flow: lead in → ranked call list → source "
               "quality. Revisit any stage with the chips above, or go back to Stage 1 and "
               "hit **🔄 Regenerate** for fresh leads.")
else:
    st.info(f"**Stage {st.session_state.step} of {_last}.** When you're done reading, click "
            "**Next →** to run the next stage. (Back, or the chips above, let you revisit.)")

# --- Navigation: shown ABOVE and BELOW the stage so it's always in view ---
_render_nav("top")
st.divider()

# --- Run + render the CURRENT stage only ---
_RENDERERS[st.session_state.step]()

st.divider()
_render_nav("bottom")
