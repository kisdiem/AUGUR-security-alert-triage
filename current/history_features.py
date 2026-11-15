"""AIT activity-relative summaries and the selected AUGUR contrast view."""
import numpy as np


def profile(summary):
    """Use the same four pooled blocks and 27 action types as the sealed experiment."""
    activity = summary[:, :16].reshape(-1, 4, 4)[:, :, 1]
    actions = summary[:, 16:].reshape(-1, 4, 27)
    return (actions - activity[:, :, None]).reshape(-1, 108)


def views(host, entity):
    """Build all three history views from two aligned 935-dimensional inputs."""
    if host.ndim != 2 or entity.shape != host.shape or host.shape[1] != 935:
        raise ValueError("Expected aligned host and entity matrices with 935 columns.")
    if not np.array_equal(host[:, :811], entity[:, :811]):
        raise ValueError("Current-session inputs differ between host and entity.")
    hp = profile(host[:, 811:])
    ep = profile(entity[:, 811:])
    h = np.c_[host[:, :827], hp]
    return {
        "host_profile": h,
        "entity_only_profile": np.c_[entity[:, :827], ep],
        "profile_contrast": np.c_[h, entity[:, 811:827], ep - hp],
    }

