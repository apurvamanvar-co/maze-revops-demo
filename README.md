# Maze RevOps — Lead Prioritization (prototype)

A guided, step-by-step Streamlit demo that scores inbound leads **before anyone
contacts them**, surfaces the few worth an SDR's time, routes them to reps, and
shows which marketing sources produce the best leads.

## Design philosophy (the point of the prototype)

- **The base score is deterministic, rules-based Python** on structured fields —
  cheap, fast, and fully explainable. No LLM touches it.
- **Claude (Sonnet) is used only on the one unstructured field a rule can't read**
  — the lead's free-text "what are you trying to do?" goal from the capture form —
  and only for leads that already clear the warm threshold.

That rules-vs-AI boundary is deliberate: it shows where AI adds leverage versus
where it's overkill.

Only **pre-touch** lead data is used (HubSpot capture/behavior, Clay enrichment,
Product free-tier usage). No Outreach / Gong / Salesforce — a sequence reply, a
recorded call, or an open opportunity all mean the lead has already been worked,
which is downstream of this step.

## The walkthrough

Demo notes → Data flood → Rules score signals → Claude reads intent →
Tier & route → SDR worklist → Source → lead quality.

## Run locally

```bash
pip install -r requirements.txt
# optional — without it, "why now" uses a deterministic rules fallback:
#   macOS/Linux:  export ANTHROPIC_API_KEY=sk-ant-...
#   Windows:      setx ANTHROPIC_API_KEY "sk-ant-..."   (then open a new terminal)
streamlit run app.py
```

The Anthropic key is read from the `ANTHROPIC_API_KEY` environment variable
(on Streamlit Community Cloud, set it under **Secrets**). It is never committed.

Synthetic data only — **no real customer data**.
