"""Re-pool registered JSON and audit counterfactuals using exact 4/2/2 IDFs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RELEASE = BASE/"ait_augur_action_contrast_20261115_v1"
D = BASE/"ait_augur_expression_cf_20261113_v1"
QUOTE = BASE/"security_paper_v9_20261030_v1/p2"
sys.path.insert(0,str(BASE/"augur_cf_transfer_20261111_v4"))
from pair_inference import read, pinned, read_pickle, normalizers, pick_source, original_weight, encode, pool, token_weights, write, sha

OUT = ROOT/"edits"
FAMILIES = ("json","quote")


def prepare(frozen):
    c = frozen.c
    cfg = read(D/"input_pins.json")
    trial = read(D/"trial_protocol.json")
    gate = read(pinned(QUOTE/"audits/final_quote_gate.json",trial["quote_gate_sha256"]))
    legacy = Path(cfg["legacy"])
    archive = Path(cfg["rotation"])
    manifest = read(pinned(D/"source_scan.json",cfg["scan_sha256"]))
    pins = {"plan":sha(ROOT/"plan.json"),"prepare":sha(Path(__file__)),
            "source":sha(D/"input_pins.json"),"gate":trial["quote_gate_sha256"],
            "rotation":sha(c.f.ORIG422/"results/protocol.json")}
    ready = OUT/"complete.json"
    if ready.is_file():
        lock = read(ready)
        assert lock["pins"] == pins
        for name,hashcode in lock["files"].items():
            pinned(OUT/name,hashcode)
        return lock
    OUT.mkdir(exist_ok=True)
    pl, deps, constants = normalizers(legacy)
    assert deps == manifest["dependencies"] and constants == manifest["constant_literals_sha256"]
    def collect(s):
        return pick_source(archive/"cache/scenes"/s["scene"]/"sources.sqlite",
                           s,pl,pl.__globals__["normalize"])
    with ThreadPoolExecutor(max_workers=3) as queue:
        json_edits={sid:row for d in queue.map(collect,manifest["scenes"]) for sid,row in d.items() if sid in c.f.L}
    assert len(json_edits)==257
    norm = pl.__globals__["normalize"]
    quote_edits={}
    for sid,row in gate["selected_one_event"].items():
        if sid not in c.f.L:
            continue
        assert row["adapter_equal"] and row["adapter_checkpoint_equal"] and row["dns43_single_event_equal"]
        assert all(row["checks"].values())
        quote_edits[sid] = dict(id=row["event_id"],sig=norm(row["signature"]),
                                before=row["native_old_raw"], after=row["native_new_raw"])
    assert len(quote_edits)==230
    edits = {"json":json_edits,"quote":quote_edits}
    keys=sorted({row[f] for family in edits.values() for row in family.values() for f in ("before","after")})
    sigs = {i:v["signatures"] for i,v in c.f.r.COHORT.items()}
    raw_data=read_pickle(legacy/"raw/sessions.pkl",cfg["bank_pins"])
    files=cfg["bank_pins"]
    original_texts=read(pinned(legacy/"raw/texts.json", files[str(legacy/"raw/texts.json")]))
    vectors=np.load(pinned(legacy/"raw/vectors.npy",files[str(legacy/"raw/vectors.npy")]),mmap_mode="r",allow_pickle=False)
    wants=set(keys)
    index={}
    for i,t in enumerate(original_texts):
        if t in wants:
            index.setdefault(t,i)
    del original_texts
    model=legacy/"model"
    for path,h in files.items():
        if str(model)+"/" in path:
            pinned(Path(path),h)
    args=SimpleNamespace(encoder_python=Path("<RESEARCH_ROOT>/reproductions/joint_darpa_20261018/env/bin/python"),
                         encoder_pythonpath="<RESEARCH_ROOT>/reproductions/nllog_crossscene_422_20261023/deps",
                         out=OUT,threads=4,allow_compatible_encoder=False)
    lookup,runtime=encode(model,keys,args,cfg["encoder_runtime"])
    before_error=float(max(np.max(np.abs(lookup[r["before"]]-vectors[index[r["before"]]]))
                           for family in edits.values() for r in family.values()))
    assert before_error <= 2e-5, before_error
    added={}
    def vec(i):
        return vectors[i] if i<len(vectors) else added[i]
    for k in keys:
        if k not in index:
            ix=len(vectors)+len(added)
            index[k]=ix
            added[ix]=lookup[k]
    del lookup
    embed_changes={}
    for fam,rows in edits.items():
        embed_changes[fam]=min(float(np.max(np.abs(vec(index[r["after"]])-vec(index[r["before"]])))) for r in rows.values())
    weight=original_weight(legacy)
    files_out={}
    summaries={}
    for fold, split in c.f.original_422["folds"].items():
        idf=c.f.TRAIN_422[fold][-1]
        roles=c.f.r.role_indices(split)
        cohort=set().union(*(set(c.f.r.IDS[roles[role]]) for role in ("train","validation","test")))
        assert len(cohort)==11482
        for fam,all_rows in edits.items():
            which=tuple(sorted(set(all_rows)&cohort))
            natural, alternate=[], []
            for sid in which:
                r=all_rows[sid]
                pairs=raw_data[sid]["pairs"]
                sign=sigs[sid]
                counts=Counter()
                for (s,i),n in pairs.items():
                    counts[s]+=n
                assert dict(counts)==dict(sign), sid
                w,den=token_weights(sign,idf,weight)
                natural.append(pool(pairs,w,den,vec))
                alternate.append(pool(pairs,w,den,vec,((r["sig"],index[r["before"]]),(r["sig"],index[r["after"]]))))
            p=OUT/(fold+"_"+fam+".npz")
            natural,alternate=np.stack(natural),np.stack(alternate)
            live=np.any(natural!=alternate,axis=1)
            c.npz(p,ids=np.array(which)[live],original=natural[live],alternate=alternate[live])
            files_out[p.name]=sha(p)
            summaries[p.name]={"parents":int(live.sum()),"unchanged_after_pooling":int((~live).sum())}
        print("prepared",fold,flush=True)
    lock={"pins":pins,"files":files_out,"source_parents":{k:len(v) for k,v in edits.items()},
          "fold_pairs":summaries,"maximum_encoder_original_delta":before_error,
          "minimum_embedding_changes":embed_changes,"runtime":runtime}
    write(ready,lock)
    return lock

