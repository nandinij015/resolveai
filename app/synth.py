"""Synthetic claim generator with known ground truth (injected defects -> expected finding codes)."""
import random
from datetime import date, timedelta
from .models import Claim, Document

START = date(2026, 1, 1)
DESC = {"theft": ["Laptop and camera stolen from parked vehicle", "Bicycle taken from locked garage"],
        "collision": ["Rear-ended at traffic signal", "Side impact at junction"],
        "fire": ["Kitchen fire from electrical fault", "Smoke and fire damage to living room"],
        "water_damage": ["Pipe burst flooded the bathroom", "Roof leak soaked bedroom ceiling"],
        "earthquake": ["Cracked walls after tremor"]}
EXCL = ["Gradual rust and wear and tear on the unit", "Damage was intentional, caused by the policyholder",
        "Property used for business purposes when loss occurred", "Unit had been deteriorating slowly for years"]
NAMES = ["Laptop", "Camera", "Phone", "Repair labour", "Parts", "Flooring", "Paint", "Appliance"]
VERIFIED = {"AMT_MISMATCH", "SUM_MISMATCH", "INV_DATE", "POLICE_BEFORE_LOSS", "PERIL_MISMATCH"}
POLICY_CODES = {"POLICY_ID", "OUT_OF_PERIOD", "LATE_REPORT", "NOT_COVERED", "EXCLUSION", "OVER_LIMIT", "MISSING_DOC"}
RISK = {"RISK_EARLY_LOSS", "RISK_NEAR_LIMIT"}

def _items(rng, target):
    a = round(target * 0.5 / 10) * 10; b = round(target * 0.3 / 10) * 10
    return list(zip(rng.sample(NAMES, 3), [a, b, target - a - b]))

def _two_peril(s): return s["peril"] in ("theft", "collision")

def i_amt(r, s): s["claimed_delta"] = r.choice([-1, 1]) * r.randrange(100, 1000, 50); s["codes"].add("AMT_MISMATCH"); return True
def i_sum(r, s): s["printed_delta"] = r.randrange(100, 600, 50); s["codes"].add("SUM_MISMATCH"); return True
def i_near(r, s): s["items"] = _items(r, r.randrange(9000, 9900, 10)); s["codes"].add("RISK_NEAR_LIMIT"); return True
def i_over(r, s): s["items"] = _items(r, r.randrange(10500, 12000, 10)); s["codes"].add("OVER_LIMIT"); return True
def i_late(r, s): s["rep_off"] = r.randint(31, 60); s["codes"].add("LATE_REPORT"); return True
def i_oop(r, s): s["loss"] = date(2025, r.randint(9, 12), 10); s["codes"].add("OUT_OF_PERIOD"); return True
def i_early(r, s): s["loss"] = START + timedelta(days=r.randint(2, 12)); s["codes"].add("RISK_EARLY_LOSS"); return True
def i_invdate(r, s):
    s["inv_off"] = r.randint(3, 30) if s["peril"] == "theft" else -r.randint(5, 30); s["codes"].add("INV_DATE"); return True
def i_pbl(r, s):
    if not _two_peril(s) or "police_report" in s["drop"]: return False
    s["pol_off"] = -r.randint(1, 5); s["codes"].add("POLICE_BEFORE_LOSS"); return True
def i_missing(r, s):
    if not _two_peril(s): return False
    s["drop"].add(r.choice(["police_report"] + (["photo_report"] if s["peril"] == "collision" else []))); s["codes"].add("MISSING_DOC"); return True
def i_perilmm(r, s):
    if not _two_peril(s): return False
    s["incident"] = "fire"; s["codes"].add("PERIL_MISMATCH"); return True
def i_excl(r, s): s["desc"] = r.choice(EXCL); s["codes"].add("EXCLUSION"); return True
def i_wrongpol(r, s): s["policy_no"] = "POL-999"; s["codes"].add("POLICY_ID"); return True
def i_notcov(r, s):
    s["peril"], s["desc"] = "earthquake", DESC["earthquake"][0]; s["inv_off"] = r.randint(3, 15); s["codes"].add("NOT_COVERED"); return True

GROUPS = [{"notcov": i_notcov, "excl": i_excl, "wrongpol": i_wrongpol},
          {"missing": i_missing, "perilmm": i_perilmm},
          {"amt": i_amt, "sum": i_sum, "near": i_near, "over": i_over},
          {"late": i_late, "oop": i_oop, "early": i_early, "invdate": i_invdate, "pbl": i_pbl}]
ALL = {k: f for g in GROUPS for k, f in g.items()}

def _money(r, v, fancy): return f"${v:,.2f}" if fancy else f"{v:.2f}"

def make_claim(rng: random.Random, claim_id: str, injectors: list[str] | None = None):
    peril = rng.choices(["theft", "collision", "fire", "water_damage"], [3, 2, 2, 2])[0]
    s = {"peril": peril, "loss": date(2026, 2, 1) + timedelta(days=rng.randrange(0, 300)), "rep_off": rng.randint(1, 20),
         "inv_off": -rng.randint(30, 200) if peril == "theft" else rng.randint(3, 15), "pol_off": rng.randint(0, 2),
         "items": _items(rng, rng.randrange(1500, 7500, 50)), "claimed_delta": 0, "printed_delta": 0, "policy_no": "POL-001",
         "desc": rng.choice(DESC[peril]), "incident": peril, "drop": set(), "codes": set()}
    if injectors is None:
        n = rng.choices([0, 1, 2], [3, 5, 2])[0]
        injectors = [rng.choice(list(g)) for g in rng.sample(GROUPS, n)]
    order = list(ALL)
    for name in sorted(injectors, key=order.index): ALL[name](rng, s)  # fixed order: peril-changing injectors first
    s["incident"] = s["incident"] if s["incident"] != s["peril"] or "PERIL_MISMATCH" in s["codes"] else s["peril"]
    loss = s["loss"]; ddmm = rng.random() < 0.3; fancy = rng.random() < 0.5
    D = lambda d: d.strftime("%d/%m/%Y") if ddmm else d.isoformat()
    total = float(sum(a for _, a in s["items"])); printed = total + s["printed_delta"]; claimed = printed + s["claimed_delta"]
    rep, inv, pol = loss + timedelta(days=s["rep_off"]), loss + timedelta(days=s["inv_off"]), loss + timedelta(days=s["pol_off"])
    peril_txt = s["peril"].replace("_", " ")
    docs = [Document(doc_id=f"{claim_id}-form", doc_type="claim_form", text="\n".join([
        "CLAIM FORM", f"{rng.choice(['Policy Number', 'Policy No'])}: {s['policy_no']}",
        f"{rng.choice(['Date of Loss', 'Loss Date'])}: {D(loss)}", f"{rng.choice(['Date Reported', 'Reported On'])}: {D(rep)}",
        f"Peril: {peril_txt.title() if rng.random() < .5 else peril_txt}", f"{rng.choice(['Claimed Amount', 'Amount Claimed'])}: {_money(rng, claimed, fancy)}",
        f"Description: {s['desc']}"])),
        Document(doc_id=f"{claim_id}-inv", doc_type="invoice", text="\n".join(
            ["INVOICE INV-" + claim_id[-4:], f"Invoice Date: {D(inv)}"] + [f"Item: {n} | {_money(rng, a, fancy)}" for n, a in s["items"]] +
            [f"Total: {_money(rng, printed, fancy)}"]))]
    if _two_peril(s) and "police_report" not in s["drop"]:
        docs.append(Document(doc_id=f"{claim_id}-police", doc_type="police_report", text="\n".join(
            ["POLICE REPORT", f"Report Date: {D(pol)}", f"Incident: {s['incident'].replace('_', ' ')}", "Reference: PR-" + claim_id[-4:]])))
    if s["peril"] == "collision" and "photo_report" not in s["drop"]:
        docs.append(Document(doc_id=f"{claim_id}-photo", doc_type="photo_report", text="PHOTO REPORT\nImages: 6"))
    truth_facts = {"policy_number": s["policy_no"], "date_of_loss": loss.isoformat(), "date_reported": rep.isoformat(), "peril": s["peril"],
                   "claimed_amount": f"{claimed:.2f}", "invoice_date": inv.isoformat(), "invoice_total": f"{printed:.2f}",
                   "line_items": sorted(f"{n}|{a:.2f}" for n, a in s["items"])}
    if any(d.doc_type == "police_report" for d in docs):
        truth_facts.update(report_date=pol.isoformat(), incident=s["incident"])
    codes = set(s["codes"])
    rec = ("flag_inconsistencies" if codes & (VERIFIED | POLICY_CODES - {"MISSING_DOC"}) else
           "request_more_evidence" if "MISSING_DOC" in codes else "approve_for_review")
    return Claim(claim_id=claim_id, documents=docs), {"facts": truth_facts, "codes": codes, "recommendation": rec}

SAMPLES = {"Clean claim": ["notcov"][:0], "Amount mismatch": ["amt"], "Missing police report": ["missing"], "Early-loss risk signal": ["early"]}
def sample_claims():
    out = []
    for i, (label, inj) in enumerate(SAMPLES.items()):
        rng = random.Random(100 + i)
        while True:
            c, t = make_claim(rng, f"CLM-{9001 + i}", inj)
            if inj or True:
                if not inj or t["codes"]: break
        out.append({"label": label, "claim": c.model_dump()})
    return out
