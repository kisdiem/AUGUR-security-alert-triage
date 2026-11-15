"""Bounded, role-aware state for observations associated by the AIT adapter."""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

FAMILIES = (
    ("exec", "proc_spawn", "process_exit", "execution_arguments_observation"),
    ("file_read", "object_read", "object_open"),
    ("file_write", "file_create", "file_delete", "object_write", "object_create", "object_delete"),
    ("net_connect", "net_send", "net_receive"),
    ("dns_query", "dns_forward", "dns_observation"),
    ("http_request",),
    ("authentication", "session_start", "session_end", "account_observation"),
    ("audit_path_observation", "cwd_observation"),
)
INDEX = {action: i for i, names in enumerate(FAMILIES) for action in names}
ROLES = ("actor", "target", "account")
ROLE_WIDTH = len(ROLES) * (len(FAMILIES) + 1)
LINK_NAMES = (
    "same_directed_peer", "same_peer_and_family", "same_recent_peer", "same_recent_peer_and_family",
    "target_last_actor_is_current", "target_last_family_actor_is_current",
    "read_or_execute_after_same_actor_write", "target_prior_successful_execution_by_actor",
    "actor_parent_matches_observation", "actor_account_matches_observation",
    "account_fail_then_observed_success", "spawn_matches_prior_reported_parent",
)
WIDTH = ROLE_WIDTH + len(LINK_NAMES)
RECENT = 6 * 3600 * 10**9
WRITES = frozenset(("file_write", "object_write", "file_create", "object_create"))
READ_EXEC = frozenset(("file_read", "object_read", "object_open", "exec"))


@dataclass(slots=True)
class Trace:
    peer: str
    time: int


@dataclass(slots=True)
class State:
    families: Counter = field(default_factory=Counter)
    total: int = 0
    links: dict = field(default_factory=dict)

    def update(self, family):
        self.families[family] += 1
        self.total += 1


def matches(trace, peer):
    return trace is not None and peer is not None and trace.peer == peer


def recent(trace, peer, now):
    return matches(trace, peer) and now - trace.time <= RECENT


class EntityHistory:
    def __init__(self):
        self.states = {}
        self.last_ns = None

    def before(self, event):
        now = event["timestamp_ns"]
        if self.last_ns is not None and now < self.last_ns:
            raise ValueError("Observation precedes current state")
        out = np.zeros(WIDTH, dtype=np.float32)
        family = INDEX.get(event["action"])
        a, b, account = (event.get(role) for role in ROLES)
        for i, (role, key) in enumerate(zip(ROLES, (a, b, account), strict=True)):
            state = self.states.get((role, key))
            if state is None:
                continue
            o = i * (len(FAMILIES) + 1)
            # A low-support history tends toward zero without exporting total or exact age.
            denominator = state.total + 2
            for action, count in state.families.items():
                out[o + action] = count / denominator
            if family is not None:
                out[o + len(FAMILIES)] = state.families[family] / denominator

        actor = self.states.get(("actor", a))
        target = self.states.get(("target", b))
        user = self.states.get(("account", account))
        sa, st, su = (v.links if v is not None else {} for v in (actor, target, user))
        peer, same = sa.get("out"), sa.get(("out", family))
        action, success = event["action"], event.get("outcome") == "success"
        spawned = self.states.get(("actor", b)) if action == "proc_spawn" and b else None
        out[ROLE_WIDTH:] = (
            matches(peer, b), matches(same, b), recent(peer, b, now), recent(same, b, now),
            matches(st.get("in"), a), matches(st.get(("in", family)), a),
            action in READ_EXEC and matches(st.get("writer"), a),
            action == "exec" and matches(st.get("executor"), a),
            matches(sa.get("parent"), event.get("parent")), matches(sa.get("account"), account),
            action == "authentication" and success and recent(su.get("failed"), event.get("host"), now),
            spawned is not None and matches(spawned.links.get("parent"), a),
        )
        return out

    def apply(self, event):
        self.last_ns = now = event["timestamp_ns"]
        action = event["action"]
        family = INDEX.get(action)
        a, b, account = (event.get(role) for role in ROLES)
        present = {}
        for role, key in zip(ROLES, (a, b, account), strict=True):
            if key is None:
                continue
            role_key = (role, key)
            state = self.states.get(role_key)
            if state is None:
                state = self.states[role_key] = State()
            present[role] = state.links
            if family is not None:
                state.update(family)
        if a:
            links = present["actor"]
            for label, peer in (("out", b), ("parent", event.get("parent")), ("account", account)):
                if peer:
                    links[label] = Trace(peer, now)
            if b and family is not None:
                links["out", family] = links["out"]
        if a and b:
            links = present["target"]
            links["in"] = Trace(a, now)
            if family is not None:
                links["in", family] = links["in"]
            if event.get("outcome") == "success":
                if action in WRITES:
                    links["writer"] = links["in"]
                if action == "exec":
                    links["executor"] = links["in"]
        if account and action == "authentication" and event.get("outcome") == "failure" and event.get("host"):
            present["account"]["failed"] = Trace(event["host"], now)

