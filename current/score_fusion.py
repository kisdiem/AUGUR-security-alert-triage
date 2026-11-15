"""Probability-scale score fusion for the selected AUGUR configuration."""
import numpy as np


def agreement(old, host, direct, role, relation):
    """Take a direct-entity adjustment when both available scorers agree in sign."""
    host_delta = host - old
    direct_delta = direct - old
    beta = np.where(relation, .35, np.where(role, .20, 0.))
    same_direction = (host_delta * direct_delta) > 0
    adjustment = .20 * host_delta + np.where(same_direction, beta * direct_delta, 0.)
    return old + adjustment

