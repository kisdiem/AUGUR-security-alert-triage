"""Reconstruct typed JSON expression pairs against frozen raw session pooling."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/"augur_cf_transfer_20261111_v4"))
from pair_inference import read, pinned, read_pickle, normalizers, pick_source, original_weight, encode, pool, token_weights, write

def main():
    cfg=read(HERE/"input_pins.json")
    legacy=Path(cfg["legacy"])
    rotation=Path(cfg["rotation"])
    hashes=cfg["bank_pins"]
    source=read(pinned(HERE/"source_scan.json",cfg["scan_sha256"]))
    normalize,deps,constants=normalizers(legacy)
    assert deps==source["dependencies"] and constants==source["constant_literals_sha256"]
    scan={x["scene"]:x for x in source["scenes"]}
    def pick(scene):
        got=pick_source(rotation/"cache/scenes"/scene/"sources.sqlite",scan[scene],
                        normalize,normalize.__globals__["normalize"])
        print("JSON selection",scene,len(got),flush=True)
        return got
    with ThreadPoolExecutor(max_workers=3) as p:
        choices={sid:r for s in p.map(pick,cfg["scenes"]) for sid,r in s.items()}
    write(HERE/"json_selection.json",{s:sha256(v["id"].encode()).hexdigest() for s,v in choices.items()})
    raw=read_pickle(legacy/"raw/sessions.pkl",hashes)
    sessions=read_pickle(Path(cfg["crossscene"])/"cache/sessions.pkl",hashes)
    signatures={r["id"]:r["signatures"] for items in sessions.values() for r in items}
    del sessions
    assert set(raw)==set(signatures)
    choices={sid:e for sid,e in choices.items() if sid in signatures}
    assert len(choices)==257
    texts=sorted({row[k] for row in choices.values() for k in ("before","after")})
    need=set(texts)
    bankpath=pinned(legacy/"raw/texts.json",hashes[str(legacy/"raw/texts.json")])
    old=read(bankpath)
    bank={s:i for i,s in enumerate(old) if s in need}
    vecfile=legacy/"raw/vectors.npy"
    actual=np.load(pinned(vecfile,hashes[str(vecfile)]),mmap_mode="r",allow_pickle=False)
    assert len(actual)==len(old)
    del old
    model=legacy/"model"
    for path,digest in hashes.items():
        if str(model)+"/" in path:
            pinned(Path(path),digest)
    args=SimpleNamespace(encoder_python=Path("<RESEARCH_ROOT>/reproductions/joint_darpa_20261018/env/bin/python"),
          encoder_pythonpath="<RESEARCH_ROOT>/reproductions/nllog_crossscene_422_20261023/deps",
          out=HERE,threads=4,allow_compatible_encoder=False)
    vectors,runtime=encode(model,texts,args,cfg["encoder_runtime"])
    pairs={(s,bank[row["before"]]):row for s,row in choices.items()}
    errors=[float(np.max(np.abs(vectors[r["before"]]-actual[i]))) for (_,i),r in pairs.items()]
    assert max(errors)<=2e-5,max(errors)
    extra=[]
    for text in texts:
        if text not in bank:
            bank[text]=len(actual)+len(extra)
            extra.append(vectors[text])
    def vector(i):
        return actual[i] if i<len(actual) else extra[i-len(actual)]
    weight=original_weight(legacy)
    out=HERE/"json_views"
    out.mkdir(exist_ok=False)
    counts={}
    for fold in cfg["folds"]:
        source_train=rotation/"data/folds"/fold/"train_validation.npz"
        source_test=Path(cfg["heldout"])/fold/"test.npz"
        idf=cfg["idfs"][fold]
        current_idf=read(source_train.parent/"idf_original.json")
        assert current_idf==idf,fold
        for role,path in (("train",source_train),("test",source_test)):
            rec=read(Path(cfg["raw_model_archive"])/fold/("validation_receipt.json" if role=="train" else "complete.json"))
            with np.load(pinned(path,rec["source_sha256"]),allow_pickle=False) as a:
                suffix="_train" if role=="train" else ("_test" if "ids_test" in a else "")
                ids=a["ids"+suffix].astype(str)
                natural=a["X_S0"+suffix]
            use=np.flatnonzero(np.isin(ids,list(choices)))
            replaced=[]
            for i in use:
                sid=ids[i]
                pair=raw[sid]["pairs"]
                sigs=signatures[sid]
                count=Counter()
                for (k,_),n in pair.items():
                    count[k]+=n
                assert dict(count)==sigs
                w,denom=token_weights(sigs,idf,weight)
                np.testing.assert_array_equal(pool(pair,w,denom,vector),natural[i,384:768],err_msg=sid)
                row=choices[sid]
                edit=((row["sig"],bank[row["before"]]),(row["sig"],bank[row["after"]]))
                replaced.append(pool(pair,w,denom,vector,edit))
            views=np.stack(replaced)
            assert np.all(np.any(views!=natural[use,384:768],axis=1))
            out_file=out/f"{fold}_{role}.npz"
            np.savez_compressed(out_file,ids=ids[use],positions=use,raw=views)
            counts[f"{fold}/{role}"]=dict(n=len(use),sha256=sha256(out_file.read_bytes()).hexdigest())
    write(HERE/"json_ready.json",dict(counts=counts,source_parents=len(choices),
          encoder_runtime=runtime,raw_parity=True,max_original_vector_error=max(errors)))
    print("JSON pairs ready for eight folds",flush=True)

if __name__=="__main__":
    main()

