# 🔍 Deep Search Agentic System

A single-file multi-agent research system built with the **OpenAI Agents SDK** and **Google Gemini**, using **Tavily** for web search.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────┐
│ RequirementGatheringAgent│  Clarifies intent, expands vague terms
└──────────┬───────────────┘
           │ handoff
           ▼
┌──────────────────────────┐
│      PlanningAgent       │  Breaks query into 3-5 sub-queries
└──────────┬───────────────┘
           │ handoff
           ▼
┌──────────────────────────┐
│  LeadAgent (Orchestrator)│ ◄────────────────────────────────────────┐
└──────┬────────┬──────────┘                                          │
       │        │                                                      │
       │   as_tool                                                     │
       │        ├──► SearchAgent ──► web_search (Tavily) ─────────────┤
       │        │         └──► extract_page (Tavily) ─────────────────┤
       │        │                                                      │
       │        ├──► ReflectionAgent  (reviews gaps, quality) ────────┤
       │        │                                                      │
       │        └──► CitationsAgent   (formats all sources) ──────────┘
       │
       └──► Final markdown report
```

### Agent Roles

| Agent | Role |
|---|---|
| `RequirementGatheringAgent` | Understands and clarifies user intent |
| `PlanningAgent` | Creates a structured research plan |
| `LeadAgent` | Orchestrates all sub-agents, writes final report |
| `SearchAgent` | Searches + extracts web content (has tools) |
| `ReflectionAgent` | Reviews research quality, identifies gaps |
| `CitationsAgent` | Formats and validates all sources |

### Patterns Used

| Pattern | Where |
|---|---|
| **Tool Calling** | `SearchAgent` calls `web_search` and `extract_page` |
| **Agent as Tool** | `SearchAgent`, `ReflectionAgent`, `CitationsAgent` used via `.as_tool()` |
| **Handoffs** | `RequirementGathering` → `Planning` → `LeadAgent` |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/deep-search-agent.git
cd deep-search-agent

# 2. Install
pip install -r requirements.txt

# 3. Add API keys
cp .env.example .env
# Edit .env — add GEMINI_API_KEY and TAVILY_API_KEY

# 4. Run
python main.py "What are the latest breakthroughs in quantum computing?"
```

Get your keys:
- Gemini → https://aistudio.google.com/apikey  (free tier available)
- Tavily → https://app.tavily.com

---

## Usage

```bash
python main.py "<your research query>"
```

## Why Gemini with OpenAI Agents SDK?

Google exposes Gemini via an OpenAI-compatible REST API. The SDK lets you plug in any compatible client:

```python
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(gemini_client)
```

This means you get all the power of the Agents SDK (tools, handoffs, agents-as-tools) with Gemini models.

---

## Tech Stack

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [Google Gemini](https://aistudio.google.com) via OpenAI-compatible API
- [Tavily](https://tavily.com) — AI-optimized web search & extraction
- Python 3.11+
