"""Re-encode a single audited DNS suffix presentation per AIT parent."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import re
import sqlite3
import sys

import numpy as np

ROOT = Path("<RESEARCH_ROOT>/reproductions/security_completion_20261026_v1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from rotation_features import DEFAULT_BASE, DEFAULT_OLD, Dependencies, VectorBank

ALIASES = ("com", "net", "org", "info")
OLD = {"biz": "cloud", "cloud": "biz"}
SIG = re.compile(r"(\.)(biz|cloud)(\s+tld\b)", re.I)

def rank(s):
    return hashlib.sha256(("ait-suffix-v1/42/" + s).encode()).hexdigest()

def render_signature(text, suffix):
    rendered, count = SIG.subn(lambda m: m[1] + suffix + m[3], text)
    if count != 1:
        raise ValueError(f"Unexpected suffix signature {text!r}")
    return rendered

def render_raw(text, name, suffix):
    pattern = re.compile(re.escape(name), re.I)
    raw, count = pattern.subn(lambda m: m[0].rsplit(".", 1)[0] + "." + suffix, text)
    return SIG.sub(lambda m: m[1] + suffix + m[3], raw), count

def read_scene(scene):
    import gzip
    with gzip.open(ROOT / "cache/scenes" / scene / "rows.pkl.gz", "rb") as f:
        return pickle.load(f)

def read_events(scene, ids):
    path = ROOT / "cache/scenes" / scene / "sources.sqlite"
    result = {}
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        for sid in ids:
            result[sid] = list(db.execute(
                "SELECT id,record,raw_id,cf_audit FROM events "
                "WHERE sid=? AND cf_record IS NOT NULL ORDER BY t,id", (sid,)))
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    manifest = json.loads((a.folder / "request.json").read_text())
    dep = Dependencies(DEFAULT_BASE)
    bank = VectorBank(dep, Path(DEFAULT_OLD), ROOT / "cache/embeddings", threads=4)
    # Access archived bank only through lookup/get. New embeddings stay in this run.
    natives, raw_rows, event_lookup = {}, {}, {}
    selected = set(manifest["candidates"]) | set(manifest["validation_ids"])
    for scene in sorted({sid.rsplit("_", 1)[0] for sid in selected}):
        scene_ids = {sid for sid in selected if sid.startswith(scene + "_")}
        obj = read_scene(scene)
        natives.update({sid:r for sid,r in obj["native_rows"].items()
                        if sid in scene_ids or sid in manifest["training_ids"]})
        raw_rows.update({sid:obj["raw_rows"][sid]["pairs"] for sid in scene_ids})
        event_lookup.update(read_events(scene, scene_ids))
        del obj
        print("read scene", scene, len(scene_ids), flush=True)
    assert set(manifest["training_ids"]) <= natives.keys()
    idf = dep.fit_idf([natives[sid] for sid in manifest["training_ids"]])
    required = {raw_id for events in event_lookup.values() for _,_,raw_id,_ in events if raw_id >= 0}
    reverse = {i:text for text,i in bank.lookup.items() if i in required}
    assert required == reverse.keys()

    candidates = {}
    rejected = Counter()
    for sid, events in event_lookup.items():
        pairs = raw_rows[sid]
        for eid,payload,raw_id,check in sorted(events, key=lambda e:rank(e[0])):
            if raw_id < 0:
                continue
            audit = json.loads(check)
            if not (audit["eligible"] and audit["changed"]):
                continue
            name = dep.normalize(audit["old_name"])
            original_suffix = audit["old_suffix"]
            if original_suffix not in OLD:
                raise ValueError("Unexpected registered original suffix")
            signature = dep.normalize(json.loads(payload)["text"])
            text = reverse[raw_id]
            _, count = render_raw(text, name, ALIASES[0])
            if not count:
                rejected[(manifest['candidates'].get(sid, 'validation'), 'render_raw_no_name')] += 1
                continue
            if not any(s == signature and bank.q_to_bank[i] == raw_id and n > 0
                       for (s,i),n in pairs.items() if i >= 0):
                rejected[(manifest['candidates'].get(sid, 'validation'), 'raw_pair_unmatched')] += 1
                continue
            renderings = tuple((render_signature(signature, suffix),render_raw(text, name, suffix)[0])
                               for suffix in (OLD[original_suffix], *ALIASES))
            assert all(raw != text for _,raw in renderings)
            candidates[sid] = (eid, original_suffix, signature, raw_id, renderings)
            break
    train = [sorted((sid for sid,y in manifest["candidates"].items() if y==label and sid in candidates),
                    key=rank) for label in (0,1)]
    print('candidate exclusions', dict(rejected), 'valid', len(candidates), flush=True)
    budget = min(map(len, train))
    if budget < 20:
        raise ValueError(f"Insufficient auditable parents per class: {list(map(len, train))}")
    train_ids = train[0][:budget] + train[1][:budget]
    valid_ids = sorted(set(manifest["validation_ids"]) & candidates.keys(),key=rank)[:64]
    ids = train_ids + valid_ids
    text_to_index = {}
    for sid in ids:
        for pair in candidates[sid][4]:
            for text in pair:
                if text not in text_to_index:
                    text_to_index[text] = len(text_to_index)
    print("parents",len(train_ids),"paired validation",len(valid_ids),"vectors",len(text_to_index),flush=True)
    if not valid_ids:
        raise ValueError("No paired validation rows")

    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(4)
    modeldir = dep.Q / "model"
    tokenizer = AutoTokenizer.from_pretrained(modeldir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(modeldir, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True).eval().requires_grad_(False)
    texts = list(text_to_index)
    embedded = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = tokenizer(texts[start:start+32],padding=True,truncation=True,max_length=256,return_tensors="pt")
            values = model(**batch).last_hidden_state
            mask = batch["attention_mask"][...,None]
            embedded.append(((values*mask).sum(1)/mask.sum(1).clamp_min(1)).numpy())
    embedded = np.concatenate(embedded)
    assert np.isfinite(embedded).all()

    @lru_cache
    def old_vector(idx):
        # Keep no derived vectors on the old path.
        return bank.get(idx)

    def base_features(sid):
        signatures = natives[sid]["signatures"]
        weights = {s:dep.weight(s,idf) for s in signatures}
        denom = sum(weights[s]*n for s,n in signatures.items())
        if abs(denom)<1e-8:
            weights = {s:1. for s in signatures}
            denom = sum(signatures.values())
        denom += 1e-8
        alert = sum((old_vector(bank.lookup[s])*(weights[s]*n) for s,n in signatures.items()),
                    np.zeros(384,np.float64))/denom
        raw = sum((old_vector(int(bank.q_to_bank[i]))*(weights[s]*n)
                   for (s,i),n in raw_rows[sid].items() if i>=0), np.zeros(384,np.float64))/denom
        return np.r_[alert,raw],weights,denom

    baseline, variants, records = [],[],[]
    for sid in ids:
        eid, old, signature, raw_id, renderings = candidates[sid]
        base,weights,denom = base_features(sid)
        delta_weight = weights[signature]/denom
        source = np.r_[old_vector(bank.lookup[signature]),old_vector(raw_id)]
        rendered = np.stack([np.r_[embedded[text_to_index[s]],embedded[text_to_index[r]]]
                             for s,r in renderings])
        variants.append((base + delta_weight*(rendered-source)).astype(np.float32))
        baseline.append(base.astype(np.float32))
        records.append(dict(id=sid,event=eid,source_suffix=old,target_suffixes=(OLD[old],*ALIASES),
                            split="train" if sid in train_ids else "validation",pooling_weight=delta_weight))
    arrays = dict(ids=np.array(ids),train_size=len(train_ids),original=np.stack(baseline),variants=np.stack(variants))
    np.savez_compressed(a.folder/"views.npz",**arrays)
    audit = dict(records=records,train_per_class=budget,validation_pairs=len(valid_ids),
                 validated_original_label_source=True,only_current_alert_and_raw_views_changed=True,
                 native_dns_and_history_unchanged=True,source_rows_unchanged=True,
                 rule_ids_not_synthesized=True,one_alert_per_parent=True,
                 old_paired_views=1,new_views_per_parent=len(ALIASES),
                 edited_feature_rows=int(np.any(arrays["original"][:,None]!=arrays["variants"],axis=-1).sum()),
                 actual_network_replay=False,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.folder/"render_audit.json").write_text(json.dumps(audit,indent=2))
    print("features ready", arrays["variants"].shape,flush=True)

if __name__=="__main__":
    main()


