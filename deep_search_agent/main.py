"""
Deep Search Agentic System
==========================
Architecture (from diagram):
  RequirementGatheringAgent
        │ handoff
        ▼
  PlanningAgent
        │ handoff
        ▼
  LeadAgent (Orchestrator)
        ├──► SearchAgent (as_tool) ──► web_search tool (Tavily)
        │         └──────────────────────────────────────────► back to Lead
        ├──► ReflectionAgent (as_tool) ◄──────────────────────────────────►
        └──► CitationsAgent  (as_tool) ◄──────────────────────────────────►

Concepts demonstrated:
  1. Tool Calling    — SearchAgent calls Tavily web_search via @function_tool
  2. Agent as Tool   — Search / Reflection / Citations agents used as tools by Lead
  3. Handoffs        — RequirementGathering → Planning → Lead
          
Note on Gemini + OpenAI Agents SDK:
  The SDK supports any OpenAI-compatible endpoint.
  Google exposes Gemini via an OpenAI-compatible API at:
  https://generativelanguage.googleapis.com/v1beta/openai/
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv
import json
from tavily import TavilyClient
from agents import Agent, Runner, function_tool, handoff, AsyncOpenAI, OpenAIChatCompletionsModel, set_default_openai_client, set_tracing_disabled

load_dotenv(find_dotenv())
set_tracing_disabled(True)
# ════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Gemini model to use — flash is fast & cheap, pro is more capable
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

MAX_CONTENT_LEN = 3000  # chars per page snippet


# ════════════════════════════════════════════════════════════════════════
# GEMINI CLIENT SETUP (OpenAI-compatible endpoint)
# ════════════════════════════════════════════════════════════════════════

gemini_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

# Make it the default so all agents use Gemini automatically
set_default_openai_client(gemini_client)

def gemini(model: str = GEMINI_MODEL) -> OpenAIChatCompletionsModel:
    """Helper: returns a Gemini model instance for use in agents."""
    return OpenAIChatCompletionsModel(model=model, openai_client=gemini_client)


# ════════════════════════════════════════════════════════════════════════
# TOOL — web_search  (used by SearchAgent)
# ════════════════════════════════════════════════════════════════════════

@function_tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily and return results as JSON.
    Each result has: title, url, content, score.
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
    )
    results = [
        {
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "content": r.get("content", "")[:MAX_CONTENT_LEN],
            "score":   round(r.get("score", 0.0), 3),
        }
        for r in response.get("results", [])
    ]
    return json.dumps(results, ensure_ascii=False)


@function_tool
def extract_page(url: str) -> str:
    """
    Extract the full readable content of a URL using Tavily Extract.
    Returns JSON: { url, content, success }
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        response = client.extract(urls=[url])
        items = response.get("results", [])
        if items:
            raw = items[0].get("raw_content", "")[:MAX_CONTENT_LEN]
            return json.dumps({"url": url, "content": raw, "success": True})
        return json.dumps({"url": url, "content": "", "success": False})
    except Exception as e:
        return json.dumps({"url": url, "content": "", "success": False, "error": str(e)})


# ════════════════════════════════════════════════════════════════════════
# SEARCH AGENT  — has web_search tool, used as_tool by LeadAgent
# ════════════════════════════════════════════════════════════════════════

search_agent = Agent(
    name="SearchAgent",
    model=gemini(),
    instructions="""
You are a focused web research agent. Given a sub-query:

1. Call `web_search` with the sub-query.
2. Pick the top 2 most relevant URLs from the results.
3. Call `extract_page` for each URL to get richer content.
4. Return a concise JSON summary:
   {
     "sub_query": "<the query>",
     "sources": [
       {
         "title": "...",
         "url": "...",
         "key_points": ["point 1", "point 2", "point 3"]
       }
     ]
   }

Only include facts found in the search results. Be precise and concise.
""",
    tools=[web_search, extract_page],
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION AGENT — reviews findings, sends quality feedback to Lead
# ════════════════════════════════════════════════════════════════════════

reflection_agent = Agent(
    name="ReflectionAgent",
    model=gemini(),
    instructions="""
You are a critical research reviewer. You receive research findings from the Lead Agent.

Your job:
1. Check if the findings are comprehensive and cover all angles of the query.
2. Identify any gaps, contradictions, or weak sources.
3. Suggest 1-3 additional sub-queries if important angles are missing.
4. Rate the research quality: poor / fair / good / excellent.

Return a JSON object:
{
  "quality": "good",
  "gaps": ["gap 1", "gap 2"],
  "suggested_queries": ["follow-up query 1"],
  "feedback": "Overall the research is solid but missing X..."
}

Be constructive and specific. If quality is excellent, say so and return empty gaps.
""",
)


# ════════════════════════════════════════════════════════════════════════
# CITATIONS AGENT — formats and validates all source citations
# ════════════════════════════════════════════════════════════════════════

citations_agent = Agent(
    name="CitationsAgent",
    model=gemini(),
    instructions="""
You are a citations formatting specialist. You receive research findings with URLs.

Your job:
1. Extract all URLs mentioned in the findings.
2. Format them as clean citations with "n", "url", and "reliable" fields.
3. Remove any duplicate URLs.
4. Flag any URLs that look unreliable.

Return a JSON object:
{
  "citations": [
    {"n": 1, "url": "https://...", "title": "...", "reliable": true},
    {"n": 2, "url": "https://...", "title": "...", "reliable": false}
  ],
  "total": 5
}
""",
)


# ════════════════════════════════════════════════════════════════════════
# LEAD AGENT (Orchestrator)
#   — uses Search / Reflection / Citations agents as tools
#   — receives handoff from PlanningAgent
# ════════════════════════════════════════════════════════════════════════

lead_agent = Agent(
    name="LeadAgent",
    model=gemini(),
    instructions="""
You are the Lead Research Orchestrator. You receive a structured research plan.

Your workflow:
1. Read the plan and identify all sub-queries to research.
2. For EACH sub-query, call `SearchAgent` to gather findings.
3. Collect all findings, then call `ReflectionAgent` to review quality.
4. If ReflectionAgent suggests follow-up queries, research those too via `SearchAgent`.
5. Call `CitationsAgent` to format all sources.
6. Return a final JSON object exactly in this format:
{
  "query_display": "<original query>",
  "sources_count": <total unique sources>,
  "sub_queries_count": <number of sub-queries run>,
  "time_s": <estimate of time taken in seconds>,
  "quality": "<quality rating from ReflectionAgent>",
  "sources": [{"title": "...", "reliable": true}],
  "report": "<The full markdown research report with executive summary, themes, and takeaways>",
  "reflection_feedback": "<The feedback string from ReflectionAgent>",
  "citations": [{"n": 1, "url": "...", "reliable": true}]
}

Ensure the "report" field contains the full markdown. Do not include any text outside this JSON object.
""",
    tools=[
        search_agent.as_tool(
            tool_name="SearchAgent",
            tool_description="Research a focused sub-query. Returns key findings and source URLs.",
        ),
        reflection_agent.as_tool(
            tool_name="ReflectionAgent",
            tool_description="Review research findings for quality, gaps, and suggested follow-up queries.",
        ),
        citations_agent.as_tool(
            tool_name="CitationsAgent",
            tool_description="Format and validate all source URLs into clean numbered citations.",
        ),
    ],
)


# ════════════════════════════════════════════════════════════════════════
# PLANNING AGENT — breaks user query into a research plan, hands off to Lead
# ════════════════════════════════════════════════════════════════════════

planning_agent = Agent(
    name="PlanningAgent",
    model=gemini(),
    instructions="""
You are a research planning specialist. You receive a clarified research request.

Your job:
1. Analyze the request and identify the core research objective.
2. Break it into 3-5 focused sub-queries covering different angles.
3. Define the expected output format and depth.
4. Hand off the structured plan to the LeadAgent.

Format your handoff message as:
## Research Plan
**Objective:** <clear one-line objective>
**Sub-queries:**
1. <sub-query 1>
2. <sub-query 2>
3. <sub-query 3>
**Expected depth:** <brief / standard / deep>
**Output format:** markdown

Then immediately hand off to LeadAgent.
""",
    handoffs=[handoff(lead_agent)],
)


# ════════════════════════════════════════════════════════════════════════
# REQUIREMENT GATHERING AGENT — clarifies user intent, hands off to Planning
# ════════════════════════════════════════════════════════════════════════

requirement_gathering_agent = Agent(
    name="RequirementGatheringAgent",
    model=gemini(),
    instructions="""
You are a research requirements specialist. You receive a raw user query.

Your job:
1. Analyze the query to understand the user's real intent.
2. Identify: topic, scope, desired depth, any specific angles mentioned.
3. Rewrite it as a clear, structured research request — expanding any vague terms.
4. Do NOT ask follow-up questions. Work with what you have.
5. Hand off the clarified request to PlanningAgent immediately.

Format your handoff:
## Clarified Research Request
**Original query:** <user's raw query>
**Interpreted intent:** <what the user actually wants>
**Scope:** <narrow / medium / broad>
**Key aspects to cover:** <aspect 1>, <aspect 2>, <aspect 3>

Then hand off to PlanningAgent.
""",
    handoffs=[handoff(planning_agent)],
)


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

async def run(query: str) -> str:          # ← add return type
    if not GEMINI_API_KEY:
        return "Error: Missing GEMINI_API_KEY"
    if not TAVILY_API_KEY:
        return "Error: Missing TAVILY_API_KEY"

    print(f"\n Deep Search Agentic System")
    print(f"{'─' * 55}")
    print(f"Query : {query}")
    print(f"Model : {GEMINI_MODEL}")
    print(f"{'─' * 55}")
    print(f"\n Starting pipeline: RequirementGathering → Planning → Lead\n")

    result = await Runner.run(
        starting_agent=requirement_gathering_agent,
        input=query,
    )
    

    print("\n" + "═" * 55)
    print(result.final_output)
    print("═" * 55 + "\n")
    return result.final_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deep Search Agentic System")
    parser.add_argument("query", type=str, help="Your research query")
    args = parser.parse_args()
    asyncio.run(run(args.query))
