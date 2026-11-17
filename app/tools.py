"""Deterministic tools behind the ADK agents. Dates, sums, limits and clause matching are computed
here, never by the LLM, so every finding is reproducible and carries evidence ids."""
from __future__ import annotations
import json, math, re
from collections import Counter
from datetime import date
from pathlib import Path
from .models import Assessment, Claim, Edge, Fact, Finding, Node

ROOT = Path(__file__).resolve().parent.parent
POLICY = json.loads((ROOT / "data" / "policy.json").read_text())
CLAUSES = {c["id"]: c for c in POLICY["clauses"]}

# ---------------- Document Agent: extraction ----------------
FIELDS = {
    "claim_form": {"policy number": "policy_number", "policy no": "policy_number", "date of loss": "date_of_loss",
                   "loss date": "date_of_loss", "date reported": "date_reported", "reported on": "date_reported",
                   "peril": "peril", "cause of loss": "peril", "claimed amount": "claimed_amount",
                   "amount claimed": "claimed_amount", "description": "description"},
    "invoice": {"invoice date": "invoice_date", "date of invoice": "invoice_date", "total": "invoice_total",
                "amount due": "invoice_total"},
    "police_report": {"report date": "report_date", "date of report": "report_date", "incident": "incident",
                      "incident type": "incident"},
}
LINE = re.compile(r"^\s*([A-Za-z ]+?)\s*:\s*(.+?)\s*$")
ITEM = re.compile(r"^\s*Item\s*:\s*(.+?)\s*\|\s*\$?([\d,]+(?:\.\d+)?)\s*$", re.I)

def norm_money(s: str) -> str: return f"{float(re.sub(r'[^\d.]', '', s)):.2f}"
def norm_peril(s: str) -> str: return re.sub(r"[\s-]+", "_", s.strip().lower())
def norm_date(s: str) -> str:
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})$", s.strip())  # DD/MM/YYYY
    return f"{m[3]}-{m[2]}-{m[1]}" if m else s.strip()
def _d(s):
    try: return date.fromisoformat(s)
    except Exception: return None

def _parse_line(doc_type, line):
    m = ITEM.match(line)
    if m and doc_type == "invoice": return "line_item", f"{m[1].strip()}|{norm_money(m[2])}"
    m = LINE.match(line)
    if not m: return None
    name = FIELDS.get(doc_type, {}).get(m[1].strip().lower())
    if not name: return None
    v = m[2].strip()
    if name.endswith(("amount", "total")): v = norm_money(v)
    elif name.startswith("date") or name.endswith("_date"): v = norm_date(v)
    elif name in ("peril", "incident"): v = norm_peril(v)
    return name, v

def extract_facts(claim: Claim) -> list[Fact]:
    facts = []
    for d in claim.documents:
        for i, line in enumerate(d.text.splitlines()):
            try: r = _parse_line(d.doc_type, line)
            except ValueError: r = None
            if r: facts.append(Fact(fact_id=f"fact:{d.doc_id}:{i}", name=r[0], value=r[1], doc_id=d.doc_id, passage=line.strip()))
    return facts

# ---------------- Policy Agent: retrieval over clauses ----------------
STOP = set("the a an of and or to in on for is are was were by with from as at be that this while loss excluded".split())
def stem(w):
    for suf in ("ing", "ed", "ly", "es", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3: return w[: -len(suf)]
    return w
def toks(s): return [stem(w) for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP]

class Retriever:
    """Lexical TF-IDF retriever. Swap `search` for an embedding / pgvector backend without touching callers."""
    def __init__(self, clauses):
        self.clauses = clauses
        docs = [toks(c["text"]) for c in clauses]
        df = Counter(w for d in docs for w in set(d))
        self.idf = {w: math.log((1 + len(docs)) / (1 + df[w])) + 1 for w in df}
        self.vecs = [self._vec(d) for d in docs]
    def _vec(self, t):
        c = Counter(t); v = {w: n * self.idf.get(w, 1.0) for w, n in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}
    def search(self, query, kind=None, k=1):
        q = self._vec(toks(query)); out = []
        for c, v in zip(self.clauses, self.vecs):
            if kind and c["kind"] != kind: continue
            shared = set(q) & set(v)
            out.append((c, sum(q[w] * v[w] for w in shared), len(shared)))
        return sorted(out, key=lambda x: -x[1])[:k]

RETRIEVER = Retriever(POLICY["clauses"])

def mk(kind, code, title, detail, evidence, calc=None):
    ev = list(evidence) + ([f"calc:{code}"] if calc else [])
    return Finding(code=code, kind=kind, title=title, detail=detail, evidence=ev, calc=calc)

def _one(facts): return lambda n: next((f for f in facts if f.name == n), None)

def check_policy(claim: Claim, facts: list[Fact]) -> list[Finding]:
    out, one, p = [], _one(facts), POLICY
    loss, rep, peril, desc, num, claimed = (one(n) for n in ("date_of_loss", "date_reported", "peril", "description", "policy_number", "claimed_amount"))
    ps, pe = _d(p["period"]["start"]), _d(p["period"]["end"])
    if num and num.value != p["policy_id"]:
        out.append(mk("policy_issue", "POLICY_ID", "Policy number does not match the policy on file",
                      f"Claim cites {num.value}; policy on file is {p['policy_id']}.", [num.fact_id, "clause:C-1.1"]))
    if loss and _d(loss.value) and not ps <= _d(loss.value) <= pe:
        out.append(mk("policy_issue", "OUT_OF_PERIOD", "Date of loss is outside the policy period",
                      f"Loss on {loss.value}; policy runs {ps} to {pe}.", [loss.fact_id, "clause:C-1.1"]))
    if loss and rep and _d(loss.value) and _d(rep.value):
        days = (_d(rep.value) - _d(loss.value)).days
        if days > p["report_within_days"]:
            out.append(mk("policy_issue", "LATE_REPORT", "Loss reported after the reporting window",
                          f"Reported {days} days after loss; limit is {p['report_within_days']}.", [loss.fact_id, rep.fact_id, "clause:C-5.1"],
                          {"expr": f"{rep.value} - {loss.value} (days)", "result": days}))
    if peril and peril.value not in CLAUSES["C-2.1"]["perils"]:
        out.append(mk("policy_issue", "NOT_COVERED", "Peril is not a covered peril",
                      f"'{peril.value}' is not listed in the covered perils.", [peril.fact_id, "clause:C-2.1"]))
    if desc:
        for c, score, shared in RETRIEVER.search(desc.value, kind="exclusion"):
            if shared >= 2 and score >= 0.2:
                out.append(mk("policy_issue", "EXCLUSION", f"Description matches exclusion {c['id']}",
                              f"Description overlaps with clause {c['id']} (similarity {score:.2f}). Human review required.", [desc.fact_id, f"clause:{c['id']}"]))
    if claimed and float(claimed.value) > p["limit"]:
        out.append(mk("policy_issue", "OVER_LIMIT", "Claimed amount exceeds the per-claim limit",
                      f"Claimed {claimed.value}; limit {p['limit']}.", [claimed.fact_id, "clause:C-4.1"],
                      {"expr": f"{claimed.value} - {p['limit']}", "result": round(float(claimed.value) - p["limit"], 2)}))
    if peril:
        req = next((c for c in p["clauses"] if c["kind"] == "documents" and c["peril"] == peril.value), None)
        if req:
            have = {d.doc_type for d in claim.documents}
            missing = [r for r in req["required"] if r not in have]
            if missing:
                out.append(mk("missing_evidence", "MISSING_DOC", "Required supporting document is missing",
                              f"Clause {req['id']} requires: {', '.join(missing)} not found.", [f"clause:{req['id']}", f"claim:{claim.claim_id}"]))
    return out

# ---------------- Consistency Agent ----------------
def check_consistency(claim: Claim, facts: list[Fact]) -> list[Finding]:
    out, one = [], _one(facts)
    claimed, total = one("claimed_amount"), one("invoice_total")
    items = [f for f in facts if f.name == "line_item"]
    loss, inv, pol, peril, inc = one("date_of_loss"), one("invoice_date"), one("report_date"), one("peril"), one("incident")
    if claimed and total and claimed.value != total.value:
        diff = round(float(claimed.value) - float(total.value), 2)
        out.append(mk("verified_discrepancy", "AMT_MISMATCH", "Claim form amount differs from invoice total",
                      f"Claim form {claimed.value} vs invoice total {total.value} (difference {diff:+.2f}).",
                      [claimed.fact_id, total.fact_id], {"expr": f"{claimed.value} - {total.value}", "result": diff}))
    if items and total:
        s = round(sum(float(i.value.split("|")[1]) for i in items), 2)
        if abs(s - float(total.value)) > 0.005:
            out.append(mk("verified_discrepancy", "SUM_MISMATCH", "Invoice line items do not add up to the invoice total",
                          f"Line items sum to {s:.2f}; invoice states {total.value}.", [i.fact_id for i in items] + [total.fact_id],
                          {"expr": " + ".join(i.value.split("|")[1] for i in items), "result": s}))
    if loss and inv and peril and _d(loss.value) and _d(inv.value):
        gap = (_d(inv.value) - _d(loss.value)).days
        if peril.value == "theft" and gap > 0:
            out.append(mk("verified_discrepancy", "INV_DATE", "Ownership invoice is dated after the reported theft",
                          f"Invoice dated {gap} days after loss.", [inv.fact_id, loss.fact_id], {"expr": f"{inv.value} - {loss.value} (days)", "result": gap}))
        if peril.value != "theft" and gap < 0:
            out.append(mk("verified_discrepancy", "INV_DATE", "Repair invoice is dated before the reported loss",
                          f"Invoice dated {-gap} days before loss.", [inv.fact_id, loss.fact_id], {"expr": f"{inv.value} - {loss.value} (days)", "result": gap}))
    if loss and pol and _d(loss.value) and _d(pol.value) and _d(pol.value) < _d(loss.value):
        out.append(mk("verified_discrepancy", "POLICE_BEFORE_LOSS", "Police report predates the reported loss",
                      f"Report dated {pol.value}; loss dated {loss.value}.", [pol.fact_id, loss.fact_id]))
    if peril and inc and peril.value != inc.value:
        out.append(mk("verified_discrepancy", "PERIL_MISMATCH", "Police incident type differs from claimed peril",
                      f"Claim says {peril.value}; police report says {inc.value}.", [peril.fact_id, inc.fact_id]))
    # advisory risk signals: NOT proof of fraud
    ps, lim = _d(POLICY["period"]["start"]), POLICY["limit"]
    if loss and _d(loss.value) and 0 <= (_d(loss.value) - ps).days <= 14:
        out.append(mk("suspected_risk_signal", "RISK_EARLY_LOSS", "Loss occurred within 14 days of policy start",
                      "Advisory only. Early losses are common and innocent; this is not evidence of fraud.", [loss.fact_id, "clause:C-1.1"]))
    if claimed and 0.9 * lim <= float(claimed.value) <= lim:
        out.append(mk("suspected_risk_signal", "RISK_NEAR_LIMIT", "Claimed amount is within 10% of the policy limit",
                      "Advisory only. Not evidence of fraud.", [claimed.fact_id, "clause:C-4.1"]))
    return out

# ---------------- Review Agent: recommendation, graph, guardrails ----------------
def recommend(findings: list[Finding]) -> str:
    kinds = {f.kind for f in findings}
    if kinds & {"verified_discrepancy", "policy_issue"}: return "flag_inconsistencies"
    if "missing_evidence" in kinds: return "request_more_evidence"
    return "approve_for_review"

def template_rationale(rec, findings):
    body = " ".join(f"[{f.code}] {f.title}." for f in findings) or "No discrepancies, policy issues or missing evidence were found."
    return (f"Recommendation: {rec.replace('_', ' ')}. {body} This is decision support for a human reviewer; "
            "risk signals are advisory and are not evidence of fraud. No claim is rejected automatically.")

def assemble(claim: Claim, facts: list[Fact], findings: list[Finding], rationale: str | None = None) -> Assessment:
    nodes, edges = {}, []
    def add(i, t, label, passage=None): nodes.setdefault(i, Node(id=i, type=t, label=label, passage=passage))
    add(f"claim:{claim.claim_id}", "claim", f"Claim {claim.claim_id}")
    for d in claim.documents: add(f"doc:{d.doc_id}", "document", f"{d.doc_type} ({d.doc_id})")
    for f in facts:
        add(f.fact_id, "fact", f"{f.name} = {f.value}", f.passage)
        edges.append(Edge(src=f.fact_id, dst=f"doc:{f.doc_id}", rel="extracted_from"))
    rec = recommend(findings)
    add("recommendation", "recommendation", rec.replace("_", " "))
    for fd in findings:
        for e in fd.evidence:
            if e.startswith("clause:") and e[7:] in CLAUSES: add(e, "clause", CLAUSES[e[7:]]["text"], CLAUSES[e[7:]]["text"])
            elif e.startswith("calc:") and fd.calc: add(e, "calculation", f"{fd.calc['expr']} = {fd.calc['result']}")
        add(f"finding:{fd.code}", "finding", fd.title)
        for e in fd.evidence:
            if e in nodes: edges.append(Edge(src=e, dst=f"finding:{fd.code}", rel="supports"))
        edges.append(Edge(src=f"finding:{fd.code}", dst="recommendation", rel="drives"))
    claimed = next((float(f.value) for f in facts if f.name == "claimed_amount"), 0.0)
    payable = max(0.0, min(claimed, POLICY["limit"]) - POLICY["deductible"]) if claimed else 0.0
    add("calc:payable", "calculation", f"max(0, min({claimed:.2f}, {POLICY['limit']}) - {POLICY['deductible']}) = {payable:.2f}")
    edges.append(Edge(src="calc:payable", dst="recommendation", rel="supports"))
    a = Assessment(claim_id=claim.claim_id, recommendation=rec, findings=findings, nodes=list(nodes.values()), edges=edges,
                   payable_estimate=payable, rationale=rationale or template_rationale(rec, findings))
    return validate(a)

def validate(a: Assessment) -> Assessment:
    """Guardrail: drop findings whose evidence is missing/unknown, recompute the recommendation from what survives,
    force auto_reject off, and reject rationales that cite finding codes that do not exist."""
    ids = {n.id for n in a.nodes}
    keep = [f for f in a.findings if f.evidence and all(e in ids for e in f.evidence)]
    a.dropped_unsupported = len(a.findings) - len(keep)
    a.findings, a.auto_reject = keep, False
    a.recommendation = recommend(keep)
    if not set(re.findall(r"\[([A-Z_]+)\]", a.rationale)) <= {f.code for f in keep}:
        a.rationale = template_rationale(a.recommendation, keep)
    return a

def verify_evidence(a: Assessment, claim: Claim) -> tuple[int, int]:
    """Re-check every cited evidence id against the source documents. Returns (valid, total)."""
    docs, nodes, ok, tot = {d.doc_id: d for d in claim.documents}, {n.id: n for n in a.nodes}, 0, 0
    for f in a.findings:
        for e in f.evidence:
            tot += 1; n = nodes.get(e)
            if not n: continue
            if n.type == "fact":
                d = docs.get(e.split(":")[1]); r = _parse_line(d.doc_type, n.passage) if d else None
                ok += bool(d and n.passage in d.text and r and f"{r[0]} = {r[1]}" == n.label)
            elif n.type == "clause": ok += e[7:] in CLAUSES
            elif n.type == "calculation": ok += bool(f.calc)
            else: ok += 1
    return ok, tot

def investigate(claim: Claim) -> Assessment:
    facts = extract_facts(claim)
    return assemble(claim, facts, check_policy(claim, facts) + check_consistency(claim, facts))
