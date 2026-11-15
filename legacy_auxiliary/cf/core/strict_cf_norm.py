"""Pure static strict-CF and multi-view Norm transforms for the new protocol.

No mutation, file/network access, fitting, labels, or real sensor execution.
CF eligibility is static field consistency, NOT unique-transaction certification.
Norm returns a synthetic feature view, NOT a valid Suricata sensor output.

Caller must rebuild raw fallback, DNS43, pools and historical state, fit the
separate train-only Norm IDF, and isolate CF branches from the real world.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
from typing import Any, Mapping

SCHEMA_VERSION = "strict_cf_norm_v1"
NORMAL_SUFFIX = "tld"
NORMAL_SID = "NORMALIZED_SUFFIX_INFO"
NORMAL_SIGNATURE = "ET INFO Observed DNS Query to .tld TLD"
RULES = {
    2027863: {"suffix": "biz", "rev": 4, "signature": "ET INFO Observed DNS Query to .biz TLD"},
    2027865: {"suffix": "cloud", "rev": 4, "signature": "ET INFO Observed DNS Query to .cloud TLD"},
}
SID_FOR_SUFFIX = {v["suffix"]: k for k, v in RULES.items()}
TARGET_SUFFIXES = frozenset(SID_FOR_SUFFIX)
CATEGORY = "Potentially Bad Traffic"
SEVERITY = 2
LABEL_ASSUMPTION = (
    "Equivalent controlled destination, unchanged payload and behavior intent; "
    "conditional inherited parent label, not new official truth."
)
IMMUTABLE_KEYS = frozenset({
    "id", "event_id", "parent_id", "session_id", "chain_id", "scene",
    "source_scene", "source_scenario", "family", "ordinal", "host", "host_hash", "program",
    "t", "time", "timestamp", "timestamp_ns", "@timestamp", "start", "end",
    "y", "label", "labels", "official_label", "official_labels",
    "official_attack_tags", "event_label", "time_label", "attack_tag",
    "synthetic", "synthetic_label", "synthetic_flag", "split", "role",
    "provenance", "audit",
})
SID_KEYS = frozenset({"signature_id", "suricata_sid", "sid"})
REV_KEYS = frozenset({"rev", "revision", "signature_rev"})
# Do not alter an internal payload label, e.g. file.biz.receiver.com.
SUFFIX_RE = re.compile(r"\.(?:biz|cloud)(?=\.?(?:$|[^A-Za-z0-9_.-]))", re.I)
SIGNATURE_RE = re.compile(r"ET\s+INFO\s+Observed\s+DNS\s+Query\s+to\s+\.(?:biz|cloud)\s+TLD", re.I)
FAST_SID_RE = re.compile(r"\[1:(?:2027863|2027865):4\]")
KEYED_SID_RE = re.compile(
    r"(?P<head>\b(?:signature_id|suricata_sid|sid)\b\s*[:=]\s*['\"]?)"
    r"(?P<sid>2027863|2027865)(?P<tail>['\"]?)(?!\d)", re.I)
TARGET_SID_RE = re.compile(r"(?<!\d)(?:2027863|2027865)(?!\d)")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return int(value.strip()) if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()) else None


def _typed_integer(old: Any, new: int) -> Any:
    return str(new) if isinstance(old, str) else new


def _signature(value: Any) -> str:
    value = " ".join(str(value or "").split()).lower()
    for prefix in ("suricata: alert - ", "suricata: "):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _suffix(name: Any) -> str | None:
    return name.rsplit(".", 1)[1].lower() if isinstance(name, str) and "." in name else None


def _domain_error(name: Any) -> str | None:
    if not isinstance(name, str) or not name:
        return "EMPTY_OR_NON_STRING_QUERY"
    if name != name.strip() or any(ord(c) < 33 or ord(c) > 126 for c in name):
        return "NON_ASCII_OR_WHITESPACE_QUERY_UNSUPPORTED"
    if name.endswith("."):
        return "ROOT_DOT_QUERY_UNSUPPORTED"
    labels = name.split(".")
    if len(labels) < 2 or any(not label for label in labels):
        return "EMPTY_OR_MISSING_DNS_LABEL"
    if len(name.encode("ascii")) > 253:
        return "QUERY_TOO_LONG"
    if any(len(label.encode("ascii")) > 63 for label in labels):
        return "DNS_LABEL_TOO_LONG"
    return None


def _queries(source: Mapping[str, Any]) -> tuple[list[tuple[str, dict]], list[str]]:
    data = source.get("data") or {}
    dns = data.get("dns") if isinstance(data, dict) else None
    if not isinstance(dns, dict):
        return [], []
    keys = [key for key in ("query", "queries") if dns.get(key) is not None]
    if len(keys) > 1:
        return [], ["MULTIPLE_QUERY_CONTAINERS_UNSUPPORTED"]
    if keys:
        key, value = keys[0], dns[keys[0]]
        if isinstance(value, dict):
            return [(f"data.dns.{key}", value)], []
        if isinstance(value, list):
            if any(not isinstance(v, dict) for v in value):
                return [], ["NON_OBJECT_QUERY_SLOT_UNSUPPORTED"]
            return [(f"data.dns.{key}[{i}]", v) for i, v in enumerate(value)], []
        return [], ["NON_OBJECT_QUERY_CONTAINER_UNSUPPORTED"]
    if str(dns.get("type", "")).lower() == "query" and "rrname" in dns:
        return [("data.dns", dns)], []
    return [], []


def _names(value: Any, path: str = "") -> list[tuple[str, str]]:
    out = []
    if isinstance(value, dict):
        for key, child in value.items():
            p = f"{path}.{key}" if path else str(key)
            if key in {"rrname", "query_name", "dns_query"} and isinstance(child, str):
                out.append((p, child))
            elif isinstance(child, (dict, list)):
                out.extend(_names(child, p))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            out.extend(_names(child, f"{path}[{i}]"))
    return out


def _alert(source: Mapping[str, Any]) -> dict:
    data = source.get("data") or {}
    value = data.get("alert") if isinstance(data, dict) else None
    return value if isinstance(value, dict) else {}


def _assess(source: Mapping[str, Any], record: Mapping[str, Any]) -> dict:
    alert = _alert(source)
    sid = _integer(alert.get("signature_id"))
    entries, errors = _queries(source)
    names = [entry.get("rrname") for _, entry in entries]
    description = (source.get("rule") or {}).get("description", "")
    signature_candidate = any(
        _signature(text) == _signature(rule["signature"])
        for text in (record.get("text"), alert.get("signature"), description)
        for rule in RULES.values())
    # Even a malformed/multiple container must not hide a target-name candidate.
    all_dns_names = [name for _, name in _names((source.get("data") or {}).get("dns", {}))]
    candidate = sid in RULES or signature_candidate or any(
        _suffix(n) in TARGET_SUFFIXES for n in names + all_dns_names)
    out = {"candidate": candidate, "eligible": False, "reasons": [],
           "sid": sid, "rev": _integer(alert.get("rev")),
           "query_count": len(entries), "query_names": names,
           "query_paths": [p for p, _ in entries]}
    if not candidate:
        out["reasons"] = ["NO_TARGET_CANDIDATE"]
        return out
    reasons = list(errors)
    if sid not in RULES:
        out["reasons"] = sorted(set(reasons + ["TARGET_QUERY_OR_SIGNATURE_NON_WHITELIST_SID"]))
        return out
    rule = RULES[sid]
    if _integer(alert.get("rev")) != 4:
        reasons.append("MISSING_OR_UNEXPECTED_RULE_REV")
    if alert.get("gid") is not None and _integer(alert["gid"]) != 1:
        reasons.append("UNSUPPORTED_RULE_GENERATOR_ID")
    if record.get("detector") != "Suricata":
        reasons.append("NON_SURICATA_PARSED_RECORD")
    for field, value in (("data.alert.signature", alert.get("signature")),
                         ("rule.description", description), ("record.text", record.get("text"))):
        if _signature(value) != _signature(rule["signature"]):
            reasons.append("RULE_SIGNATURE_MISMATCH:" + field)
    if len(entries) != 1:
        reasons.append("MULTI_QUERY_UNSUPPORTED" if entries else "MISSING_QUERY")
    else:
        query = entries[0][1]
        name = query.get("rrname")
        problem = _domain_error(name)
        if problem:
            reasons.append(problem)
        elif _suffix(name) != rule["suffix"]:
            reasons.append("RULE_QUERY_SUFFIX_MISMATCH")
        if query.get("rrtype") is None or not str(query["rrtype"]).strip():
            reasons.append("MISSING_QTYPE")
        if str(query.get("type", "query")).lower() != "query":
            reasons.append("NON_QUERY_TYPE_IN_QUERY_SLOT")
        for path, view in _names((source.get("data") or {}).get("dns", {}), "data.dns"):
            if isinstance(name, str) and view.lower() != name.lower():
                reasons.append("UNSYNCHRONIZED_OTHER_DNS_NAME:" + path)
        # An embedded raw JSON is another contemporaneous view, not an
        # independent license to rewrite unrelated queries. Opaque raw text
        # cannot establish that all query views refer to this one query.
        for path, raw in (("source.full_log", source.get("full_log")),
                          ("record.raw", record.get("raw"))):
            if not isinstance(raw, str) or not raw:
                continue
            decoded = _decoded(raw)
            if decoded is None:
                reasons.append("OPAQUE_RAW_VIEW_UNVERIFIED:" + path)
                continue
            for raw_path, raw_name in _names(decoded, path):
                if isinstance(name, str) and raw_name.lower() != name.lower():
                    reasons.append("RAW_QUERY_VIEW_MISMATCH:" + raw_path)
        if isinstance(name, str) and not _domain_error(name):
            suffix = "cloud" if rule["suffix"] == "biz" else "biz"
            new_name = name.rsplit(".", 1)[0] + "." + suffix
            if _domain_error(new_name):
                reasons.append("TRANSFORMED_" + str(_domain_error(new_name)))
            out.update(old_name=name, new_name=new_name, old_suffix=rule["suffix"],
                       new_suffix=suffix, new_sid=SID_FOR_SUFFIX[suffix])
    if alert.get("category") is not None and alert["category"] != CATEGORY:
        reasons.append("UNEXPECTED_RULE_CATEGORY")
    if alert.get("severity") is not None and _integer(alert["severity"]) != SEVERITY:
        reasons.append("UNEXPECTED_RULE_SEVERITY")
    out["reasons"], out["eligible"] = sorted(set(reasons)), not reasons
    return out


def _decoded(text: str) -> Any:
    if text.strip()[:1] in {"{", "["}:
        try:
            value = json.loads(text)
            if isinstance(value, (dict, list)):
                return value
        except (ValueError, TypeError):
            pass
    return None


def _cf_text(text: str, a: Mapping[str, Any]) -> str:
    # Entire JSON raw views can safely synchronize keyed IDs as well as strings.
    value = _decoded(text)
    if value is not None:
        return json.dumps(_cf_view(value, a), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    pattern = re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(a["old_name"]) + r"(?![A-Za-z0-9_.-])", re.I)
    text = pattern.sub(lambda m: m.group(0).rsplit(".", 1)[0] + "." + a["new_suffix"], text)
    text = re.sub(re.escape(RULES[a["sid"]]["signature"]), RULES[a["new_sid"]]["signature"], text, flags=re.I)
    text = re.sub(rf"\[1:{a['sid']}:4\]", f"[1:{a['new_sid']}:4]", text)
    return KEYED_SID_RE.sub(
        lambda m: m["head"] + (str(a["new_sid"]) if int(m["sid"]) == a["sid"] else m["sid"]) + m["tail"],
        text)


def _cf_view(value: Any, a: Mapping[str, Any]) -> Any:
    if isinstance(value, dict):
        own_sid = _integer(value.get("signature_id", value.get("sid")))
        out = {}
        for key, child in value.items():
            if key in IMMUTABLE_KEYS:
                out[key] = copy.deepcopy(child)
            elif key in SID_KEYS and _integer(child) == a["sid"]:
                out[key] = _typed_integer(child, a["new_sid"])
            elif key in REV_KEYS and own_sid == a["sid"]:
                out[key] = _typed_integer(child, 4)
            elif key == "category" and own_sid == a["sid"]:
                out[key] = CATEGORY
            elif key == "severity" and own_sid == a["sid"]:
                out[key] = _typed_integer(child, SEVERITY)
            else:
                out[key] = _cf_view(child, a)
        return out
    if isinstance(value, list):
        return [_cf_view(v, a) for v in value]
    if isinstance(value, tuple):
        return tuple(_cf_view(v, a) for v in value)
    return _cf_text(value, a) if isinstance(value, str) else copy.deepcopy(value)


def _norm_text(text: str) -> str:
    value = _decoded(text)
    if value is not None:
        return json.dumps(normalize_view(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = SIGNATURE_RE.sub(NORMAL_SIGNATURE, text)
    text = SUFFIX_RE.sub("." + NORMAL_SUFFIX, text)
    text = FAST_SID_RE.sub("[1:" + NORMAL_SID + ":0]", text)
    return KEYED_SID_RE.sub(lambda m: m["head"] + NORMAL_SID + m["tail"], text)


def normalize_view(value: Any) -> Any:
    """Normalize visible history/state text before encoding; not pooled vectors."""
    if isinstance(value, dict):
        own_sid = _integer(value.get("signature_id", value.get("sid")))
        out = {}
        for key, child in value.items():
            if key in IMMUTABLE_KEYS:
                out[key] = copy.deepcopy(child)
            elif key in SID_KEYS and _integer(child) in RULES:
                out[key] = NORMAL_SID
            elif key in REV_KEYS and own_sid in RULES:
                out[key] = "0" if isinstance(child, str) else 0
            elif key == "category" and own_sid in RULES:
                out[key] = CATEGORY
            elif key == "severity" and own_sid in RULES:
                out[key] = _typed_integer(child, SEVERITY)
            else:
                out[key] = normalize_view(child)
        return out
    if isinstance(value, list):
        return [normalize_view(v) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_view(v) for v in value)
    return _norm_text(value) if isinstance(value, str) else copy.deepcopy(value)


def _strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    out = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in IMMUTABLE_KEYS:
                out.extend(_strings(child, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            out.extend(_strings(child, f"{path}[{i}]"))
    elif isinstance(value, str):
        out.append((path, value))
    return out


def transform_event(source_row: Mapping[str, Any], parsed_record: Mapping[str, Any],
                    mode: str = "cf") -> dict:
    """Return independent source/record/audit; ineligible CF returns unchanged."""
    if not isinstance(source_row, Mapping) or not isinstance(parsed_record, Mapping):
        raise TypeError("source_row and parsed_record must be mappings")
    mode = {"strict_cf": "cf", "counterfactual": "cf", "normalize": "norm", "normalization": "norm"}.get(str(mode).lower(), str(mode).lower())
    if mode not in {"cf", "norm", "none"}:
        raise ValueError("mode must be cf, norm or none")
    source, record = copy.deepcopy(dict(source_row)), copy.deepcopy(dict(parsed_record))
    audit = {"schema_version": SCHEMA_VERSION, "mode": mode,
             "event_id": parsed_record.get("id"), "candidate": False,
             "eligible": False, "changed": False, "reasons": [],
             "actual_sensor_execution": False, "sensor_certified": False,
             "source_before_sha256": _digest(source_row), "record_before_sha256": _digest(parsed_record),
             "recompute": {"raw_fallback": True, "DNS43": True, "IDF_weighted_pools": True, "history_state": True}}
    if mode == "cf":
        assessment = _assess(source_row, parsed_record)
        audit.update(assessment)
        audit.update(static_only=True, label_assumption=LABEL_ASSUMPTION,
                     certification_scope="static_field_consistency_only",
                     requires_state_transition_check=True,
                     requires_isolated_parent_world=True)
        if assessment["eligible"]:
            source, record = _cf_view(source, assessment), _cf_view(record, assessment)
            before, _ = _queries(source_row); after, errors = _queries(source)
            assert not errors and len(before) == len(after) == 1
            old, new = before[0][1], after[0][1]
            assert new["rrname"] == assessment["new_name"]
            assert old["rrname"].rsplit(".", 1)[0] == new["rrname"].rsplit(".", 1)[0]
            assert {k: v for k, v in old.items() if k != "rrname"} == {k: v for k, v in new.items() if k != "rrname"}
            assert _integer(_alert(source)["signature_id"]) == assessment["new_sid"]
            residual = [path for path, text in _strings({"source": source, "record": record})
                        if assessment["old_name"].lower() in text.lower()
                        or RULES[assessment["sid"]]["signature"].lower() in text.lower()]
            if residual:
                source, record = copy.deepcopy(dict(source_row)), copy.deepcopy(dict(parsed_record))
                audit.update(eligible=False, reasons=["UNSYNCHRONIZED_TEXT_VIEW"], residual_paths=residual)
    elif mode == "norm":
        source, record = normalize_view(source), normalize_view(record)
        residual = [path for path, text in _strings({"source": source, "record": record})
                    if SUFFIX_RE.search(text) or TARGET_SID_RE.search(text)]
        audit.update(candidate=True, eligible=not residual,
                     reasons=["UNREMOVED_KNOWN_PROXY"] if residual else [],
                     feature_view_only=True, real_sensor_output=False,
                     canonical_suffix=NORMAL_SUFFIX, canonical_signature_id=NORMAL_SID,
                     residual_known_proxy_paths=residual,
                     requires_caller_history_normalization=True,
                     requires_DNS_recompute_from_transformed_source=True,
                     unremoved_or_unidentified_proxies=[
                         "Sensor family, observation coverage, workload, time and behavioral payload retained.",
                         "Physical identity/provenance retained outside features; must remain excluded.",
                         "Cached history/embeddings/DNS vectors must be rebuilt by caller.",
                         "Packet/flow byte counts and arbitrary opaque numeric IDs are not inferred or fabricated.",
                         "Lexical TLD-like non-DNS names may change; this is feature normalization, not executed behavior.",
                     ])
    else:
        audit.update(eligible=True, no_intervention=True)
    for key in IMMUTABLE_KEYS:
        if key in parsed_record:
            assert parsed_record[key] == record[key], f"immutable record field changed: {key}"
        if key in source_row:
            assert source_row[key] == source[key], f"immutable source field changed: {key}"
    audit["changed"] = source != dict(source_row) or record != dict(parsed_record)
    audit.update(source_after_sha256=_digest(source), record_after_sha256=_digest(record))
    return {"source": source, "record": record, "audit": audit}


def eligible_session(events: Any) -> dict:
    """Reject a whole parent if any change candidate fails; retain denominators."""
    events = list(events)
    changes, rejected, audits = [], [], []
    for i, event in enumerate(events):
        if isinstance(event, Mapping) and "source" in event and "record" in event:
            source, record = event["source"], event["record"]
        elif isinstance(event, (list, tuple)) and len(event) == 2:
            source, record = event
        else:
            raise TypeError("events must be {source,record,family,ordinal} mappings or pairs")
        a = transform_event(source, record, "cf")["audit"]
        audits.append(a)
        if not a["candidate"]:
            continue
        if not a["eligible"]:
            rejected.append({"event_index": i, "event_id": record.get("id"), "reasons": a["reasons"]})
        elif a["changed"]:
            changes.append({"event_index": i, "event_id": record.get("id"),
                            "old_query_name": a["old_name"], "new_query_name": a["new_name"],
                            "old_sid": a["sid"], "new_sid": a["new_sid"], "rev": 4,
                            "query_count": 1, "actual_sensor_execution": False,
                            "label_assumption": LABEL_ASSUMPTION})
    candidates = sum(a["candidate"] for a in audits)
    reasons = sorted({r for row in rejected for r in row["reasons"]})
    if not candidates:
        reasons.append("NO_TARGET_CANDIDATE_IN_PARENT")
    if candidates and not changes and not rejected:
        reasons.append("NO_ACTUAL_INPUT_CHANGE")
    return {"schema_version": SCHEMA_VERSION, "eligible": bool(changes) and not rejected,
            "reasons": reasons, "changes": changes, "events_total": len(events),
            "candidate_events": candidates, "non_candidate_events": len(events) - candidates,
            "eligible_changed_events": len(changes), "rejected_candidate_events": len(rejected),
            "rejected_events": rejected, "event_audits": audits,
            "unique_physical_transactions_confirmed": False,
            "actual_sensor_execution": False, "certification_scope": "static_field_consistency_only",
            "label_assumption": LABEL_ASSUMPTION, "requires_parent_world_replay": True}


def normalization_spec() -> dict:
    return {"schema_version": SCHEMA_VERSION, "suffixes": sorted(TARGET_SUFFIXES),
            "canonical_suffix": NORMAL_SUFFIX, "canonical_signature": NORMAL_SIGNATURE,
            "canonical_sid": NORMAL_SID, "canonical_rev": 0,
            "category_severity": "Identical actual values preserved: Potentially Bad Traffic / 2.",
            "views": ["alert text", "rule/signature/SID/rev", "raw", "structured DNS",
                      "history text", "state attributes"],
            "metadata_exclusions": sorted(IMMUTABLE_KEYS),
            "numeric_reconstruction": "Caller rebuilds DNS43 and weighted pools from transformed source.",
            "IDF": "Norm-only original-train IDF shared by Norm and Norm+repeat.",
            "scope": "Synthetic feature view; not a valid or executed sensor record."}


def _self_test() -> dict:
    def example(name="AbC.file.xlsx.receiver.biz", sid=2027863, rev=4):
        rule = RULES[sid]
        source = {"rule": {"description": "Suricata: Alert - " + rule["signature"]},
                  "data": {"alert": {"signature_id": str(sid), "rev": str(rev), "gid": "1",
                           "signature": rule["signature"], "severity": "2", "category": CATEGORY},
                           "dns": {"query": [{"rrname": name, "rrtype": "A", "type": "query", "id": 17, "tx_id": 0}]}},
                  "event_label": "dnsteal"}
        record = {"id": "immutable-event", "t": 1.0, "host": "host-ip",
                  "text": source["rule"]["description"], "raw": "", "program": None,
                  "detector": "Suricata", "location": "eve.json"}
        return source, record
    checks = []
    s, r = example(); original = copy.deepcopy((s, r)); out = transform_event(s, r, "cf")
    assert out["audit"]["eligible"] and out["audit"]["changed"]
    assert out["source"]["data"]["dns"]["query"][0]["rrname"] == "AbC.file.xlsx.receiver.cloud"
    assert out["source"]["data"]["alert"]["signature_id"] == "2027865"
    assert out["source"]["event_label"] == "dnsteal" and (s, r) == original
    checks.append("CF nonmutating; preserves payload, event identity, time, qtype and labels")
    for name, sid, rev in [("x.y.biz", 2027863, 5), ("x.y.cloud", 2027863, 4)]:
        s, r = example(name, sid, rev)
        assert not transform_event(s, r, "cf")["audit"]["eligible"]
    s, r = example(); s["data"]["dns"]["query"].append(copy.deepcopy(s["data"]["dns"]["query"][0]))
    assert "MULTI_QUERY_UNSUPPORTED" in transform_event(s, r, "cf")["audit"]["reasons"]
    checks.append("reject wrong revision, mismatched suffix and multiple queries")
    good_s, good_r = example(); s, r = example()
    s["data"]["alert"]["signature_id"] = "2029709"
    s["data"]["alert"]["signature"] = "ET HUNTING Suspicious Domain Request for Possible COVID-19 Domain M1"
    r["text"] = s["data"]["alert"]["signature"]
    parent = eligible_session([{"source": good_s, "record": good_r}, {"source": s, "record": r}])
    assert not parent["eligible"] and parent["candidate_events"] == 2
    assert len(parent["changes"]) == 1 and parent["rejected_candidate_events"] == 1
    checks.append("reject nonwhitelist rules and whole mixed parent; keep failure denominators")
    s, r = example(); raw = json.dumps({"alert": s["data"]["alert"], "dns": s["data"]["dns"]})
    s["full_log"] = raw; r["raw"] = raw
    s["history_text"] = "observed payload.receiver.cloud"
    s["state_attributes"] = {"domain": "payload.receiver.biz", "payload": "customer.biz.encoding.receiver.cloud"}
    out = transform_event(s, r, "norm")
    assert out["audit"]["eligible"]
    a = out["source"]["data"]["alert"]; decoded = json.loads(out["record"]["raw"])
    assert a["signature_id"] == NORMAL_SID and a["rev"] == "0" and a["severity"] == "2"
    assert decoded["alert"]["signature_id"] == NORMAL_SID
    assert decoded["dns"]["query"][0]["rrname"] == "AbC.file.xlsx.receiver.tld"
    assert out["source"]["state_attributes"]["payload"] == "customer.biz.encoding.receiver.tld"
    assert out["source"]["event_label"] == "dnsteal"
    checks.append("Norm syncs raw/SID/signature/DNS/history/state while retaining internal payload labels")
    assert normalize_view(normalize_view(s)) == normalize_view(s)
    assert _norm_text("foo.biz.example.com") == "foo.biz.example.com"
    assert _norm_text("foo.cloud.") == "foo.tld."
    assert _norm_text("foo.biz/path") == "foo.tld/path"
    checks.append("Norm idempotence and final-label boundaries")
    s, r = example(); s["data"]["dns"]["answer"] = [{"rrname": "different.receiver.biz"}]
    assert not transform_event(s, r, "cf")["audit"]["eligible"]
    checks.append("reject mismatched answer/alias views")
    return {"passed": True, "checks": checks, "check_count": len(checks),
            "scope": "synthetic semantic contract checks; no real data, sensor or training"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--spec", action="store_true")
    args = parser.parse_args()
    print(json.dumps(_self_test() if args.self_test else normalization_spec(), ensure_ascii=False, indent=2))

