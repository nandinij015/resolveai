"""Google ADK orchestration. LLM agents decide tool order and write the rationale; every fact, number and
citation comes from deterministic tools in tools.py and is re-validated by the guardrail before it is returned."""
import os
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types
from . import tools as T
from .models import Assessment, Claim, Fact, Finding

MODEL = os.getenv("RESOLVEAI_MODEL", "gemini-2.5-flash")
APP = "resolveai"

def extract_documents(tool_context: ToolContext) -> dict:
    """Extract structured facts, each with its source document and exact passage, from every claim document."""
    facts = T.extract_facts(Claim(**tool_context.state["claim"]))
    tool_context.state["facts"] = [f.model_dump() for f in facts]
    return {"facts_extracted": len(facts), "fields": sorted({f.name for f in facts})}

def check_policy(tool_context: ToolContext) -> dict:
    """Check coverage, exclusions, limits, reporting window and required documents against retrieved policy clauses."""
    claim, facts = Claim(**tool_context.state["claim"]), [Fact(**f) for f in tool_context.state["facts"]]
    fs = T.check_policy(claim, facts)
    tool_context.state["policy_findings"] = [f.model_dump() for f in fs]
    return {"findings": [{"code": f.code, "title": f.title} for f in fs]}

def check_consistency(tool_context: ToolContext) -> dict:
    """Cross-check documents: amounts, invoice arithmetic, date ordering, peril agreement. Also emits advisory risk signals."""
    claim, facts = Claim(**tool_context.state["claim"]), [Fact(**f) for f in tool_context.state["facts"]]
    fs = T.check_consistency(claim, facts)
    tool_context.state["consistency_findings"] = [f.model_dump() for f in fs]
    return {"findings": [{"code": f.code, "kind": f.kind, "title": f.title} for f in fs]}

def get_findings(tool_context: ToolContext) -> dict:
    """Return all findings so far (code, kind, title, detail) and the rule-based recommendation."""
    fs = [Finding(**f) for f in tool_context.state.get("policy_findings", []) + tool_context.state.get("consistency_findings", [])]
    return {"recommendation": T.recommend(fs), "findings": [f.model_dump(include={"code", "kind", "title", "detail"}) for f in fs]}

def finalize_assessment(rationale: str, tool_context: ToolContext) -> dict:
    """Build the evidence graph and final assessment. `rationale` must cite finding codes in square brackets, e.g. [AMT_MISMATCH]."""
    s = tool_context.state
    fs = [Finding(**f) for f in s.get("policy_findings", []) + s.get("consistency_findings", [])]
    a = T.assemble(Claim(**s["claim"]), [Fact(**f) for f in s["facts"]], fs, rationale)
    a.engine = "adk"; s["assessment"] = a.model_dump()
    return {"recommendation": a.recommendation, "findings": [f.code for f in a.findings]}

def _agent(name, desc, instr, tools, key):
    return LlmAgent(name=name, model=MODEL, description=desc, instruction=instr, tools=tools, output_key=key)

root_agent = SequentialAgent(name="resolveai_orchestrator", sub_agents=[
    _agent("document_agent", "Extracts claim facts and evidence.", "Call extract_documents exactly once, then reply with one short sentence.", [extract_documents], "doc_summary"),
    _agent("policy_agent", "Checks coverage, exclusions and limits.", "Call check_policy exactly once, then reply with one short sentence.", [check_policy], "policy_summary"),
    _agent("consistency_agent", "Finds contradictions and missing evidence.", "Call check_consistency exactly once, then reply with one short sentence.", [check_consistency], "consistency_summary"),
    _agent("review_agent", "Checks findings and prepares the recommendation.",
           "Call get_findings, then call finalize_assessment once with a 2-4 sentence rationale for a human claims reviewer. "
           "Cite each finding by its code in square brackets, e.g. [AMT_MISMATCH]. Treat suspected_risk_signal findings as advisory and "
           "never describe them as fraud. A discrepancy is not proof of fraud. Never recommend rejecting the claim.",
           [get_findings, finalize_assessment], "review_summary"),
])

async def investigate_adk(claim: Claim) -> Assessment:
    svc = InMemorySessionService()
    await svc.create_session(app_name=APP, user_id="reviewer", session_id=claim.claim_id, state={"claim": claim.model_dump()})
    runner = Runner(agent=root_agent, app_name=APP, session_service=svc)
    msg = types.Content(role="user", parts=[types.Part(text=f"Investigate claim {claim.claim_id}")])
    async for _ in runner.run_async(user_id="reviewer", session_id=claim.claim_id, new_message=msg): pass
    sess = await svc.get_session(app_name=APP, user_id="reviewer", session_id=claim.claim_id)
    if "assessment" not in sess.state: return T.investigate(claim)  # agent skipped finalize: fall back to deterministic path
    return Assessment(**sess.state["assessment"])
