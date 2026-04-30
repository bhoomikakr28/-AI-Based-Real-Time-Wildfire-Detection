"""
Phase 7 — Generative AI Layer
Provides three features powered by the Anthropic Claude API:
  1. Incident report generation  (after fire detection)
  2. AI-written ranger SMS alerts (replaces Twilio template)
  3. Natural language dashboard query (tool-use chat)
"""

import json
import os
import anthropic

# ── Client ─────────────────────────────────────────────────────────────────────
# Set ANTHROPIC_API_KEY in your environment before running the backend.
_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

MODEL = "claude-sonnet-4-20250514"


# ── 1. Incident Report Generation ─────────────────────────────────────────────

def generate_incident_report(detection_result: dict) -> dict:
    """
    Given a detection result dict, ask Claude to produce a structured
    JSON incident report suitable for logging / PDF export.
    """
    prompt = f"""
You are a wildfire incident reporting system.
Given this detection result, produce a structured incident report as JSON only.
No markdown, no preamble — raw JSON.

Detection result:
{json.dumps(detection_result, indent=2)}

Return a JSON object with these fields:
- incident_id       (string, e.g. "INC-2025-001")
- severity          (one of: "LOW" | "MODERATE" | "HIGH" | "CRITICAL")
- summary           (2-3 sentence plain-English summary)
- recommended_actions  (list of 3-5 action strings)
- notify_agencies   (list of relevant agencies e.g. ["Forest Department", "Fire Brigade"])
- estimated_spread_risk  (one of: "Low" | "Medium" | "High")
- report_generated_at   (ISO-8601 timestamp string)
"""
    response = _client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip possible ```json fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── 2. AI-Written Ranger Alert SMS ────────────────────────────────────────────

def generate_ranger_alert(detection_result: dict, location: str = "Unknown location") -> str:
    """
    Generate a concise, human-readable SMS alert for field rangers.
    Returns a plain string (≤160 chars ideally).
    """
    prompt = f"""
You are an emergency wildfire alert system.
Write a concise SMS alert (max 160 characters) for field rangers.
Be direct, clear, and include the key facts.

Detection:
- Decision:   {detection_result.get('decision', 'UNKNOWN')}
- Confidence: {detection_result.get('confidence', 0)}%
- Location:   {location}
- Time:       {detection_result.get('timestamp', 'unknown')}

Return ONLY the SMS text, nothing else.
"""
    response = _client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ── 3. Natural Language Dashboard Query (tool-use) ────────────────────────────

# Tool definitions Claude can call to fetch dashboard data
DASHBOARD_TOOLS = [
    {
        "name": "get_recent_detections",
        "description": "Returns the most recent fire detection events from the system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent events to return (default 5)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_alert_stats",
        "description": "Returns aggregated statistics: total detections, fire count, non-fire count, alert rate.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_system_status",
        "description": "Returns the current status of all pipeline components (YOLO, CNN, WebSocket stream, etc.).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def _handle_tool_call(tool_name: str, tool_input: dict, dashboard_data: dict) -> str:
    """Simulate tool execution by reading from in-memory dashboard_data."""
    if tool_name == "get_recent_detections":
        limit = tool_input.get("limit", 5)
        events = dashboard_data.get("recent_detections", [])[:limit]
        return json.dumps(events)
    elif tool_name == "get_alert_stats":
        return json.dumps(dashboard_data.get("stats", {}))
    elif tool_name == "get_system_status":
        return json.dumps(dashboard_data.get("system_status", {}))
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def natural_language_query(user_question: str, dashboard_data: dict) -> str:
    """
    Answer a ranger's plain-English question about the dashboard using
    Claude tool-use to fetch the right data slice.

    Parameters
    ----------
    user_question  : The ranger's free-text question
    dashboard_data : Dict with keys: recent_detections, stats, system_status

    Returns
    -------
    Plain-English reply string
    """
    messages = [{"role": "user", "content": user_question}]

    # First call — Claude decides which tools to use
    response = _client.messages.create(
        model=MODEL,
        max_tokens=1000,
        tools=DASHBOARD_TOOLS,
        messages=messages,
        system=(
            "You are a helpful wildfire monitoring assistant embedded in a ranger dashboard. "
            "Answer questions concisely and professionally. "
            "Use the available tools to fetch live data before answering."
        ),
    )

    # Agentic loop — handle tool calls until Claude produces a final text response
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_content = _handle_tool_call(block.name, block.input, dashboard_data)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_content,
                    }
                )

        # Append assistant turn + tool results and call again
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = _client.messages.create(
            model=MODEL,
            max_tokens=1000,
            tools=DASHBOARD_TOOLS,
            messages=messages,
            system=(
                "You are a helpful wildfire monitoring assistant embedded in a ranger dashboard. "
                "Answer questions concisely and professionally."
            ),
        )

    # Extract final text
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()

    return "Sorry, I couldn't generate a response."
