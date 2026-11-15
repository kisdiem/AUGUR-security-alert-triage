"""Recompute scene-balanced AIT scores from already exported per-scene metrics."""
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import csv

CSV=Path(__file__).parent/"results"/"ait_per_scene.csv"
SEEDS={42,1337,31415,2025,8675309}
SCENES={"fox","harrison","russellmitchell","santos","shaw","wardbeck","wheeler","wilson"}
by=defaultdict(list)
with CSV.open(encoding="utf-8",newline="") as f:
    for r in csv.DictReader(f):
        tag=r["protocol"]; scene=r["scene"].lower(); seed=int(r["seed"]); arm=r["arm"]
        if tag not in ("611","422") or scene not in SCENES or seed not in SEEDS:
            raise ValueError("unexpected protocol, scene, or seed")
        by[tag,scene,seed,arm].append((float(r["AP"]),float(r["Recall@1% FPR"])))
for tag in ("611","422"):
    print(f"\n{tag[0]}/{tag[1]}/{tag[2]} (eight scenes, five seeds)")
    for arm in ("000","001","010","011","100","101","110","111","current_only"):
        cells=[]
        for s in sorted(SCENES):
            for seed in sorted(SEEDS):
                rows=by[tag,s,seed,arm]
                assert len(rows)==(1 if tag=="611" else 2),(tag,s,seed,arm,len(rows))
                cells.append(tuple(fmean(x[i] for x in rows) for i in (0,1)))
        ap,rec=(100*fmean(x[i] for x in cells) for i in (0,1))
        print(f"{arm:14s} AP={ap:.2f}% Recall@1%FPR={rec:.2f}%")

