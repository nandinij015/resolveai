"""Evaluate ResolveAI on synthetic claims with known outcomes.  Usage: python -m eval.run_eval --n 500 --seed 7"""
import argparse, json, random
from pathlib import Path
from app.synth import make_claim, VERIFIED, POLICY_CODES, RISK
from app.tools import extract_facts, investigate, verify_evidence

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return round(p, 4), round(r, 4)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); rng = random.Random(a.seed)
    ex_ok = ex_tot = pol_ok = pol_tot = ev_ok = ev_tot = unsup = rec_ok = rejects = 0
    d = {"tp": 0, "fp": 0, "fn": 0}; rk = {"tp": 0, "fp": 0, "fn": 0}; clean_fp = clean_n = 0; misses = []
    for i in range(a.n):
        claim, truth = make_claim(rng, f"CLM-{i:04d}")
        facts = extract_facts(claim); by = {}
        for f in facts: by.setdefault(f.name, []).append(f.value)
        for k, v in truth["facts"].items():
            ex_tot += 1
            got = sorted(by.get("line_item", [])) if k == "line_items" else (by.get(k) or [None])[0]
            ex_ok += got == v
        res = investigate(claim); pred = {f.code for f in res.findings}
        for c in POLICY_CODES:
            pol_tot += 1; pol_ok += (c in pred) == (c in truth["codes"])
            if (c in pred) != (c in truth["codes"]): misses.append((claim.claim_id, c, c in pred))
        for c in VERIFIED:
            d["tp"] += c in pred and c in truth["codes"]; d["fp"] += c in pred and c not in truth["codes"]; d["fn"] += c not in pred and c in truth["codes"]
        for c in RISK:
            rk["tp"] += c in pred and c in truth["codes"]; rk["fp"] += c in pred and c not in truth["codes"]; rk["fn"] += c not in pred and c in truth["codes"]
        ok, tot = verify_evidence(res, claim); ev_ok += ok; ev_tot += tot
        unsup += res.dropped_unsupported > 0; rec_ok += res.recommendation == truth["recommendation"]; rejects += res.auto_reject
        if not truth["codes"]: clean_n += 1; clean_fp += bool(pred - RISK)
    n = a.n; dp, dr = prf(**d); rp, rr = prf(**rk)
    out = {"claims": n, "seed": a.seed, "extraction_accuracy": round(ex_ok / ex_tot, 4), "policy_rule_accuracy": round(pol_ok / pol_tot, 4),
           "discrepancy_precision": dp, "discrepancy_recall": dr, "risk_signal_precision": rp, "risk_signal_recall": rr,
           "evidence_citation_accuracy": round(ev_ok / ev_tot, 4), "unsupported_recommendation_rate": round(unsup / n, 4),
           "recommendation_accuracy": round(rec_ok / n, 4), "auto_reject_count": rejects,
           "clean_claim_false_flag_rate": round(clean_fp / clean_n, 4) if clean_n else 0.0, "policy_rule_errors": misses[:10]}
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
