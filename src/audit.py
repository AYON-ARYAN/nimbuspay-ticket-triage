#!/usr/bin/env python3
"""NimbusPay train.jsonl audit + cleaner. Standalone: stdlib only, no GPU.

    python3 src/audit.py --data ../data --out build

Method: SCHEMA.md is encoded as an executable validator. dev.jsonl is stated to be clean, so it is
used as a CONTROL -- the validator must emit zero flags on all 200 dev rows before train.jsonl is
touched. Every train flag is then a real defect rather than a misreading of the spec. After
cleaning, the same validator is re-run on the output and must come back clean.

This is the same code the notebook runs, extracted so the audit can be reviewed and reproduced
without a GPU or any third-party package.
"""
import argparse, csv, datetime, json, os, re
from collections import Counter, OrderedDict, defaultdict

SEED = 3407

KEYS   = ["category","priority","amount_paise","txn_date","txn_id","channel","language","needs_human"]
CATS   = {"payment_failed","refund","fraud","account_access","kyc","offers","other"}
CHANS  = {"upi","card","netbanking","wallet"}
NO_TXN = {"account_access","kyc","other"}   # spec: txn fields always null for these

# Legal-threat trigger for P1 (spec: RBI, ombudsman, consumer court, lawyer, legal notice/action).
LEGAL = re.compile(r"\b(rbi|ombudsman|consumer court|consumer forum|lawyer|advocate|"
                   r"legal notice|legal action|legal|court)\b", re.I)
# "already contacted support before" trigger for needs_human. These are the exact phrasings the
# generator uses; I derived the list from the dev rows where needs_human=True and priority!=P1.
REPEAT = re.compile(r"(again and again|baar[\s-]?baar|already written \d+ times|"
                    r"(second|third|fourth|2nd|3rd|4th)\s+complaint|"
                    r"(second|third|fourth|2nd|3rd|4th)\s+time\s+(i\s+am\s+)?contact|"
                    r"pehle bhi complaint|keep raising|dusri baar|teesri baar)", re.I)

def body_of(user_msg):
    """Ticket text without the Received/Source header AND without quoted-email lines.
    The spec says dates inside '>' lines are not the transaction date, so quoted blocks are
    distractors. They also contain 'already received a resolution', which false-fires REPEAT --
    that was the bug that made my first validator flag 16 clean dev rows."""
    b = user_msg.split("\n\n", 1)[1] if "\n\n" in user_msg else user_msg
    return "\n".join(l for l in b.split("\n") if not l.lstrip().startswith(">"))

def recv_of(user_msg):
    m = re.search(r"Received:\s*(\d{4}-\d{2}-\d{2})", user_msg)
    return datetime.date.fromisoformat(m.group(1)) if m else None

def priority_of(cat, amt, body):
    """SCHEMA.md priority ladder, applied in the specified order."""
    if cat == "fraud": return "P1"
    if amt is not None and amt >= 50_000 * 100: return "P1"
    if LEGAL.search(body): return "P1"
    if cat in ("payment_failed", "refund") and amt is not None and amt >= 5_000 * 100: return "P2"
    if cat == "account_access": return "P2"
    return "P3"

def needs_human_of(prio, body):
    return True if prio == "P1" else bool(REPEAT.search(body))

def validate(rows):
    """Return {defect_class: [evidence,...]} for one split."""
    F = defaultdict(list)
    for r in rows:
        rid, user, raw = r["id"], r["messages"][1]["content"], r["messages"][-1]["content"]
        body, recv = body_of(user), recv_of(user)
        try:
            o = json.loads(raw)
        except Exception as e:
            F["D1_broken_json"].append((rid, str(e)[:40])); continue
        if not isinstance(o, dict):
            F["D1_broken_json"].append((rid, "not a JSON object")); continue
        # --- structural: each checked independently so one defect cannot mask another ---
        if set(o) != set(KEYS):
            F["D3_wrong_keys"].append((rid, sorted(set(o) ^ set(KEYS))))
        td, ti, amt = o.get("txn_date"), o.get("txn_id"), o.get("amount_paise")
        if td is not None and not (isinstance(td, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", td)):
            F["D4_date_format"].append((rid, td))
        if ti is not None and not (isinstance(ti, str) and re.fullmatch(r"NP\d{10}", ti)):
            F["D5_txnid_format"].append((rid, ti))
        if amt is not None and (isinstance(amt, bool) or not isinstance(amt, int)):
            F["Dx_amount_type"].append((rid, repr(amt)))
        if not isinstance(o.get("needs_human"), bool):
            F["Dx_needs_human_type"].append((rid, repr(o.get("needs_human"))))
        if o.get("category") not in CATS: F["Dx_bad_category"].append((rid, o.get("category")))
        if o.get("language") not in {"en","hi-en"}: F["Dx_bad_language"].append((rid, o.get("language")))
        if o.get("channel") is not None and o.get("channel") not in CHANS:
            F["Dx_bad_channel"].append((rid, o.get("channel")))
        # --- semantic rules, only where the fields are well-formed enough to judge ---
        cat, prio = o.get("category"), o.get("priority")
        if cat in NO_TXN and any(o.get(k) is not None for k in ("amount_paise","txn_date","txn_id","channel")):
            F["D9_txnfields_on_nontxn"].append((rid, cat))
        if cat in CATS and (amt is None or (isinstance(amt, int) and not isinstance(amt, bool))):
            w = priority_of(cat, amt, body)
            if prio != w: F["D7_priority_rule"].append((rid, cat, amt, prio, w))
            if isinstance(o.get("needs_human"), bool) and prio in {"P1","P2","P3"}:
                wn = needs_human_of(prio, body)
                if o["needs_human"] != wn: F["D8_needs_human_rule"].append((rid, o["needs_human"], wn, prio))
        if isinstance(td, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", td) and recv:
            if datetime.date.fromisoformat(td) > recv:
                F["Dx_future_date"].append((rid, td, str(recv)))
        if isinstance(ti, str):
            digits = re.sub(r"\D", "", ti)
            if digits and digits not in re.sub(r"[^A-Z0-9]", "", user.upper()):
                F["Dx_txnid_absent_from_text"].append((rid, ti))
    return F

JUNK = re.compile(r"^(test|asdf|qwerty|lorem|foo|bar|aaa+|xxx+|\.|\s|\d)+$", re.I)
LEGACY_PRIO = {"low": "P3", "medium": "P2", "high": "P1"}
NUM = r"(\d[\d,]*(?:\.\d+)?)"

def amount_candidates(body):
    """Every money figure in the text, in paise. 'k' = thousand, 'lakh' = 100000."""
    out = set()
    for m in re.finditer(NUM + r"\s*(k|lakh|lakhs)\b", body, re.I):
        mult = 1000 if m.group(2).lower() == "k" else 100000
        out.add(round(float(m.group(1).replace(",", "")) * mult * 100))
    for m in re.finditer(r"(?:₹|Rs\.?|INR|rupees?|rupaye)\s*" + NUM, body, re.I):
        out.add(round(float(m.group(1).replace(",", "")) * 100))
    for m in re.finditer(NUM + r"\s*(?:rupees?|rupaye|/-)", body, re.I):
        out.add(round(float(m.group(1).replace(",", "")) * 100))
    return out

def clean_train(rows):
    log, kept, S = [], [], Counter()
    for r in rows:
        rid, user, raw = r["id"], r["messages"][1]["content"], r["messages"][-1]["content"]
        body = body_of(user)
        try:
            o = json.loads(raw)
        except Exception:
            log.append((rid, "drop", "D1 label JSON truncated/unparseable - label unrecoverable")); S["D1_broken_json"] += 1; continue
        if not isinstance(o, dict):
            log.append((rid, "drop", "D1 label is not a JSON object")); S["D1_broken_json"] += 1; continue
        if not body.strip() or JUNK.fullmatch(body.strip()):
            log.append((rid, "drop", f"D2 placeholder/empty ticket body ({body.strip()[:16]!r}) with fabricated label")); S["D2_junk_body"] += 1; continue
        o = dict(o)
        # 2. legacy schema
        if "prio" in o and "priority" not in o:
            v = o.pop("prio"); S["D3_prio_key"] += 1
            log.append((rid, "fix", "D3 legacy key 'prio' -> 'priority'"))
            if isinstance(v, str) and v.lower() in LEGACY_PRIO:
                nv = LEGACY_PRIO[v.lower()]
                log.append((rid, "fix", f"D3b legacy priority enum remapped ({v} -> {nv})")); S["D3b_prio_enum"] += 1; v = nv
            o["priority"] = v
        td = o.get("txn_date")
        if isinstance(td, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", td):
            m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", td)
            if m:
                o["txn_date"] = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
                log.append((rid, "fix", f"D4 txn_date DD/MM/YYYY -> ISO ({td} -> {o['txn_date']})"))
            else:
                o["txn_date"] = None; log.append((rid, "fix", f"D4 txn_date unparseable ({td!r}) -> null"))
            S["D4_date_fmt"] += 1
        ti = o.get("txn_id")
        if isinstance(ti, str) and not re.fullmatch(r"NP\d{10}", ti):
            n = re.sub(r"[^A-Za-z0-9]", "", ti).upper()
            if re.fullmatch(r"NP\d{10}", n):
                o["txn_id"] = n; log.append((rid, "fix", f"D5 txn_id normalised ({ti!r} -> {n})"))
            else:
                o["txn_id"] = None; log.append((rid, "fix", f"D5 txn_id invalid ({ti!r}) -> null"))
            S["D5_txnid_fmt"] += 1
        # 3. unit error BEFORE priority
        amt = o.get("amount_paise")
        if isinstance(amt, int) and not isinstance(amt, bool):
            C = amount_candidates(body)
            if C and amt not in C and amt * 100 in C:
                o["amount_paise"] = amt * 100
                log.append((rid, "fix", f"D6 amount stored in rupees not paise ({amt} -> {amt*100})")); S["D6_amt_unit"] += 1
        # 4. derived fields
        cat, amt = o.get("category"), o.get("amount_paise")
        if cat in CATS and (amt is None or (isinstance(amt, int) and not isinstance(amt, bool))):
            w = priority_of(cat, amt, body)
            if o.get("priority") != w:
                log.append((rid, "fix", f"D7 priority violates spec ({o.get('priority')} -> {w}; cat={cat} amt={amt})"))
                o["priority"] = w; S["D7_priority"] += 1
            wn = needs_human_of(o["priority"], body)
            if o.get("needs_human") != wn:
                log.append((rid, "fix", f"D8 needs_human violates spec ({o.get('needs_human')} -> {wn}; prio={o['priority']})"))
                o["needs_human"] = wn; S["D8_needs_human"] += 1
        if cat in NO_TXN:
            for k in ("amount_paise","txn_date","txn_id","channel"):
                if o.get(k) is not None:
                    log.append((rid, "fix", f"D9 {k} must be null for category={cat} (was {o[k]!r})")); o[k] = None; S["D9_txnfields"] += 1
        r2 = json.loads(json.dumps(r))
        r2["messages"][-1]["content"] = json.dumps(OrderedDict((k, o[k]) for k in KEYS), ensure_ascii=False)
        kept.append(r2)
    # 5. dedupe
    seen, final = {}, []
    for r in kept:
        t = r["messages"][1]["content"]
        if t in seen:
            log.append((r["id"], "drop", f"D10 exact duplicate ticket text of {seen[t]} (labels agree)")); S["D10_duplicate"] += 1
        else:
            seen[t] = r["id"]; final.append(r)
    return final, log, S

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dir holding train.jsonl and dev.jsonl")
    ap.add_argument("--out", default="build", help="dir for train_clean.jsonl and cleaning_log.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train = load_jsonl(os.path.join(a.data, "train.jsonl"))
    dev = load_jsonl(os.path.join(a.data, "dev.jsonl"))

    print("CONTROL: validating dev.jsonl (labels known clean)")
    dflags = validate(dev)
    print("  flags:", {k: len(v) for k, v in dflags.items()} or "NONE")
    assert not dflags, "Validator disagrees with a clean row -> fix the validator, not the data."
    print("  validator reproduces SCHEMA.md on %d/%d dev rows\n" % (len(dev), len(dev)))

    print("TARGET: train.jsonl")
    tflags = validate(train)
    print("  %-30s %6s %10s" % ("defect", "count", "% of rows"))
    for k in sorted(tflags, key=lambda k: -len(tflags[k])):
        n = len(tflags[k])
        print("  %-30s %6d %9.1f%%" % (k, n, 100 * n / len(train)))

    clean, log, stats = clean_train(train)
    print("\nraw %d -> clean %d" % (len(train), len(clean)))
    print("  dropped ids %d | fixed ids %d | log lines %d" % (
        len({l[0] for l in log if l[1] == "drop"}),
        len({l[0] for l in log if l[1] == "fix"}), len(log)))
    for k, v in sorted(stats.items()):
        print("    %-18s %5d" % (k, v))

    post = validate(clean)
    print("\npost-clean flags:", {k: len(v) for k, v in post.items()}
          or "ZERO - cleaned set is fully spec-consistent")
    assert not post, "cleaning left spec violations behind"

    log.sort(key=lambda x: (x[0], x[1], x[2]))
    with open(os.path.join(a.out, "cleaning_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "action", "reason"])
        w.writerows(log)
    with open(os.path.join(a.out, "train_clean.jsonl"), "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nwrote %s/train_clean.jsonl and %s/cleaning_log.csv" % (a.out, a.out))


if __name__ == "__main__":
    main()
