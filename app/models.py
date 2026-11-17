from typing import Literal, Optional
from pydantic import BaseModel, Field

class Document(BaseModel):
    doc_id: str
    doc_type: Literal["claim_form", "invoice", "police_report", "photo_report"]
    text: str

class Claim(BaseModel):
    claim_id: str
    documents: list[Document]

class Fact(BaseModel):
    fact_id: str
    name: str
    value: str
    doc_id: str
    passage: str

Kind = Literal["verified_discrepancy", "missing_evidence", "policy_issue", "suspected_risk_signal"]
Rec = Literal["approve_for_review", "request_more_evidence", "flag_inconsistencies"]  # no "reject", by design

class Finding(BaseModel):
    code: str
    kind: Kind
    title: str
    detail: str
    evidence: list[str] = Field(default_factory=list)
    calc: Optional[dict] = None

class Node(BaseModel):
    id: str
    type: Literal["claim", "document", "fact", "clause", "calculation", "finding", "recommendation"]
    label: str
    passage: Optional[str] = None

class Edge(BaseModel):
    src: str
    dst: str
    rel: Literal["extracted_from", "supports", "drives"]

class Assessment(BaseModel):
    claim_id: str
    recommendation: Rec
    auto_reject: bool = False
    findings: list[Finding]
    nodes: list[Node]
    edges: list[Edge]
    payable_estimate: float
    rationale: str
    dropped_unsupported: int = 0
    engine: str = "deterministic"
