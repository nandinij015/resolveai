# ResolveAI — Autonomous Insurance Claims Investigation Agent

Multi-agent system on **Google ADK + Gemini** that reconciles claim forms, invoices and police/photo reports against policy
clauses, then produces an **evidence-linked assessment for a human claims reviewer**. It never rejects a claim automatically and
keeps a *verified discrepancy* strictly separate from an advisory *suspected risk signal* (never labelled fraud).

```
Claim docs -> ADK SequentialAgent orchestrator
   Document Agent     extract facts + exact source passages
   Policy Agent       coverage / exclusions / limits / reporting window / required docs (clause retrieval)
   Consistency Agent  amount, invoice arithmetic, date-order and peril cross-checks; advisory risk signals
   Review Agent       recommendation + rationale -> Evidence Graph -> guardrail validation
-> approve_for_review | request_more_evidence | flag_inconsistencies
```

## Design decisions
- **LLM orchestrates, tools compute.** Dates, sums, limits and clause matches are deterministic Python tools (`app/tools.py`); Gemini chooses tool order and writes the rationale.
- **Evidence Graph per claim.** Nodes: claim, document, fact (with exact passage), clause, calculation, finding, recommendation. Every finding cites evidence ids; the UI shows the passage behind each one.
- **Guardrail (`validate`).** Findings with missing/unknown evidence are dropped, the recommendation is recomputed from what survives, `auto_reject` is forced false, and rationales citing non-existent finding codes are replaced.
- **No `reject` outcome exists in the type system** (`Rec` literal).

## Evaluation (synthetic, known ground truth; `python -m eval.run_eval --n 500 --seed 7`)
| Metric | Result |
|---|---|
| Extraction accuracy | 100.0% |
| Policy-rule accuracy | 99.6% |
| Discrepancy precision / recall | 100.0% / 100.0% |
| Risk-signal precision / recall | 100.0% / 100.0% |
| Evidence citation accuracy | 100.0% |
| Unsupported recommendation rate | 0.0% |
| Recommendation accuracy | 98.0% |
| Auto-rejects | 0 |

Caveat: the generator shares layout conventions with the extractor, so extraction scores are an upper bound. The known weakness is
exclusion detection on paraphrased descriptions (lexical TF-IDF retriever); the `Retriever.search` interface is the swap point for Gemini embeddings / pgvector.

## Run
```
pip install -r requirements.txt
python -m eval.run_eval --n 500
uvicorn app.main:app --reload --port 8001    # UI at http://localhost:8001
export GOOGLE_API_KEY=...                     # then choose "Google ADK + Gemini" in the UI (or ?mode=adk)
gcloud run deploy resolveai --source . --set-env-vars GOOGLE_API_KEY=...,DATABASE_URL=postgresql+psycopg2://...
```
Status: deterministic path, API, UI, eval and storage (SQLite/PostgreSQL) are tested. The ADK path (`app/agents.py`) is written against the
ADK API but was not run against live Gemini here. Document AI / Cloud Storage ingestion and real PDF parsing are not implemented.
