#!/usr/bin/env python3
"""Apply PREREGISTRATION.md section 8's machine-load rule to the time-limited arms, BEFORE any answer is opened.
Inputs only: runs/final/load_log.txt (per-minute load / MHz / package temp), runs/final/steps.log (step windows),
runs/final/predictions/MANIFEST.json (per-batch wall times of the Opus-built systems). No predictions or answers read.

An arm repeat is FLAGGED if, inside its step window:
  (a) 1-minute load > 8.0 in more than 20% of the logged minutes, or
  (b) Opus-built systems: any 300-route batch's seconds per route > 1.5 x that repeat's median over its 300-route batches, or
  (c) mean CPU MHz < 2000.
Flagged arm repeats are re-run in full after the OPUS-MODEL arm finishes; the re-run is scored.
Writes runs/final/load_check.json and .md.

  .venv/bin/python -I scripts/final_load_check.py
"""
import datetime as dt
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "runs" / "final"
LOAD_MAX, LOAD_SHARE, SLOW_X, MHZ_MIN = 8.0, 0.20, 1.5, 2000
TIME_LIMITED = re.compile(r"^(opus_run\d_rep\d(_sample)?|jpt_apply|jpt_prepare|rules)$")


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def val_load(run):
    """Descriptive (check-in 1.1): 1-minute load during the same program's VALIDATION fair-fight run, from
    runs/opus_fight_run<N>/load_log.txt (exists for runs 4-6; that log has no MHz column)."""
    p = ROOT / "runs" / f"opus_fight_run{run}" / "load_log.txt"
    if not p.exists():
        return None
    xs = [float(l.split()[1]) for l in p.read_text().splitlines() if len(l.split()) > 1 and l[:4].isdigit()]
    return {"median": st.median(xs), "p90": pct(xs, 0.9), "max": max(xs), "minutes": len(xs)} if xs else None


def main():
    steps, cur = {}, None
    for line in (FINAL / "steps.log").read_text().splitlines():
        m = re.match(r"^== (\S+) start (\S+)", line)
        if m:
            cur = m.group(1)
            steps[cur] = {"start": ts(m.group(2)), "end": None, "load_at_start": float(line.split("load ")[-1]) if "load " in line else None}
            continue
        m = re.match(r"^\s+exit (\d+) end (\S+)", line)
        if m and cur:
            steps[cur]["end"], steps[cur]["exit"] = ts(m.group(2)), int(m.group(1))
    rows = []
    lp = FINAL / "load_log.txt"
    if lp.exists():
        for line in lp.read_text().splitlines()[1:]:
            p = line.split()
            if len(p) >= 7 and p[1] != "logger":
                rows.append({"t": ts(p[0]), "load1": float(p[1]), "mhz": float(p[4]), "temp": None if p[6] == "NA" else float(p[6])})
    man = json.load(open(FINAL / "predictions" / "MANIFEST.json"))
    out = {"rule": {"load_max": LOAD_MAX, "load_share": LOAD_SHARE, "slow_x": SLOW_X, "mhz_min": MHZ_MIN},
           "logger_first_minute": rows[0]["t"].isoformat() if rows else None, "arms": {}}
    for name, s in steps.items():
        if not TIME_LIMITED.match(name) or s["end"] is None:
            continue
        w = [r for r in rows if s["start"] <= r["t"] <= s["end"]]
        a = {"start": s["start"].isoformat(), "end": s["end"].isoformat(), "minutes_logged": len(w),
             "load_at_start": s["load_at_start"], "exit": s.get("exit")}
        if w:
            a["share_load_over"] = sum(r["load1"] > LOAD_MAX for r in w) / len(w)
            a["mhz_mean"] = st.mean(r["mhz"] for r in w)
            temps = [r["temp"] for r in w if r["temp"] is not None]
            a["temp_max"] = max(temps) if temps else None
            a["load1_max"] = max(r["load1"] for r in w)
            a["load1_median"] = st.median(r["load1"] for r in w)
            a["load1_p90"] = pct([r["load1"] for r in w], 0.9)
        m_run = re.match(r"^opus_run(\d)_", name)
        if m_run:
            a["val_run_load1"] = val_load(int(m_run.group(1)))  # descriptive, not deciding
        flags = []
        if w and a["share_load_over"] > LOAD_SHARE:
            flags.append(f"(a) load > {LOAD_MAX} in {a['share_load_over']:.0%} of minutes")
        if w and a["mhz_mean"] < MHZ_MIN:
            flags.append(f"(c) mean clock {a['mhz_mean']:.0f} MHz")
        arm = name.replace("_sample", "")
        meta = (man.get(arm) or {}).get("meta") or {}
        batches = [b for b in meta.get("batches", []) if b.get("routes") == 300]
        if batches:
            per = [b["wall_s"] / b["routes"] for b in batches]
            med = st.median(per)
            a["batch_s_per_route"] = {"median": med, "max": max(per), "batches": len(per)}
            slow = [b["batch"] for b, x in zip(batches, per) if x > SLOW_X * med]
            if slow:
                flags.append(f"(b) batches {slow} slower than {SLOW_X}x median")
        if not w:
            a["note"] = "no logged minutes in this window (ran before the logger): judged by (b) and load at start"
        a["flags"] = flags
        a["FLAGGED"] = bool(flags)
        out["arms"][name] = a
    # Descriptive rows (check-in 1.3): the same-solver build, load-sensitive through the exact solver's 1 s limit on
    # large zones, like FINAL. Window = [written_utc - wall_s, written_utc] from MANIFEST. Not part of the section 8 rule.
    out["descriptive_samesolver"] = {}
    for arm, e in sorted(man.items()):
        if not arm.startswith("samesolver_run"):
            continue
        end = ts(e["written_utc"])
        start = end - dt.timedelta(seconds=e["meta"]["wall_s"])
        w = [r for r in rows if start <= r["t"] <= end]
        d = {"start": start.isoformat(), "end": end.isoformat(), "minutes_logged": len(w), "workers": e["meta"].get("workers"),
             "cpu_s_per_route_mean": e["meta"].get("cpu_s_per_route_mean")}
        if w:
            d.update(share_load_over=sum(r["load1"] > LOAD_MAX for r in w) / len(w), load1_median=st.median(r["load1"] for r in w),
                     load1_max=max(r["load1"] for r in w), mhz_mean=st.mean(r["mhz"] for r in w),
                     temp_max=max((r["temp"] for r in w if r["temp"] is not None), default=None))
        out["descriptive_samesolver"][arm] = d
    (FINAL / "load_check.json").write_text(json.dumps(out, indent=1))
    lines = ["# Machine-load check of the time-limited arms (PREREGISTRATION.md section 8)", "",
             f"Rule: flag if load1 > {LOAD_MAX} in > {LOAD_SHARE:.0%} of minutes, an Opus batch > {SLOW_X}x its median s/route, "
             f"or mean clock < {MHZ_MIN} MHz. Made by `scripts/final_load_check.py` from inputs only.", "",
             "| Step | Minutes logged | Load at start | Share load > 8 | Max load | Mean MHz | Max temp C | Batch s/route median / max | FLAGGED |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, a in out["arms"].items():
        b = a.get("batch_s_per_route")
        lines.append(f"| {n} | {a['minutes_logged']} | {a['load_at_start']} | {a.get('share_load_over', float('nan')):.0%} | "
                     f"{a.get('load1_max', '—')} | {a.get('mhz_mean', float('nan')):.0f} | {a.get('temp_max', '—')} | "
                     f"{'%.3f / %.3f' % (b['median'], b['max']) if b else '—'} | {'YES: ' + '; '.join(a['flags']) if a['FLAGGED'] else 'no'} |")
    lines += ["", "Descriptive, not deciding (check-in 1.1): 1-minute load during each final Opus-built repeat next to the same "
              "program's VALIDATION run (`runs/opus_fight_run<N>/load_log.txt`; runs 2-3 had no load log).", "",
              "| Step | Final median / p90 load | Val median / p90 / max load |", "|---|---|---|"]
    for n, a in out["arms"].items():
        if n.startswith("opus_run"):
            v = a.get("val_run_load1")
            fm = f"{a['load1_median']:.2f} / {a['load1_p90']:.2f}" if "load1_median" in a else "not logged"
            lines.append(f"| {n} | {fm} | {'%.2f / %.2f / %.2f' % (v['median'], v['p90'], v['max']) if v else 'no val load log'} |")
    lines += ["", "Caveat: the validation load logs have no CPU-clock column, so rule (c) (mean clock < 2,000 MHz) has no validation anchor."]
    if out["descriptive_samesolver"]:
        v1 = man.get("v1_exact", {}).get("meta", {})
        lines += ["", "Descriptive, not deciding (check-in 1.3): the same-solver build (`scripts/final_samesolver.py`, section 6.3). "
                  f"For comparison, FINAL (v1_exact) used {v1.get('workers')} workers and {v1.get('cpu_s_per_route_mean', float('nan')):.2f} CPU-s per route "
                  "(that includes its own zone decoding). The logger restarted about 7 minutes after this build began.", "",
                  "| Arm | Minutes logged | Workers | CPU-s/route | Share load > 8 | Median / max load | Mean MHz | Max temp C |", "|---|---|---|---|---|---|---|---|"]
        for n, d in out["descriptive_samesolver"].items():
            lines.append(f"| {n} | {d['minutes_logged']} | {d['workers']} | {d['cpu_s_per_route_mean']:.2f} | "
                         + (f"{d['share_load_over']:.0%} | {d['load1_median']:.2f} / {d['load1_max']:.2f} | {d['mhz_mean']:.0f} | {d['temp_max']} |"
                            if d["minutes_logged"] else "— | — | — | — |"))
    (FINAL / "load_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
