import numpy as np
import matplotlib.pyplot as plt

from src.array import Array
from src.failure import Failure


def draw_volts_edge_profile(failure: Failure, size_obj: float=1, unsigned: bool = False):
    volts_edge_profile = np.array(failure.volts_edge_profile.copy()).transpose() # [edge, time]
    if unsigned: np.abs(volts_edge_profile, out=volts_edge_profile)
    num_edge = volts_edge_profile.shape[0]
    num_broken_edge = len(failure.idxs_edge_broken)

    for idx_time, idx_edge_broken in enumerate(failure.idxs_edge_broken):
        volts_edge_profile[idx_edge_broken, (idx_time + 1):] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=72)

    ax.scatter(0, volts_edge_profile[failure.idxs_edge_broken[0], 0], c="red", s=size_obj, zorder=num_edge)
    for idx_edge, volt_profile in enumerate(volts_edge_profile):
        if idx_edge in failure.idxs_edge_broken:
            ax.plot(range(num_broken_edge), volt_profile, c="red", lw=size_obj, zorder=num_edge)
        else:
            ax.plot(range(num_broken_edge), volt_profile, c="black", lw=size_obj)
    
    plt.tight_layout()
    plt.show()


def draw_volts_cap_profile(failure: Failure, size_obj: float=1, unsigned: bool = False):
    volts_cap_profile = np.array(failure.volts_cap_profile.copy()).transpose() # [edge, time]
    if unsigned: np.abs(volts_cap_profile, out=volts_cap_profile)
    num_edge = volts_cap_profile.shape[0]
    num_broken_edge = len(failure.idxs_edge_broken)

    for idx_time, idx_edge_broken in enumerate(failure.idxs_edge_broken):
        volts_cap_profile[idx_edge_broken, (idx_time + 1):] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=72)

    ax.scatter(0, volts_cap_profile[failure.idxs_edge_broken[0], 0], c="red", s=size_obj, zorder=num_edge)
    for idx_edge, volt_profile in enumerate(volts_cap_profile):
        if idx_edge in failure.idxs_edge_broken:
            ax.plot(range(num_broken_edge), volt_profile, c="red", lw=size_obj, zorder=num_edge)
        else:
            ax.plot(range(num_broken_edge), volt_profile, c="black", lw=size_obj)
    
    plt.tight_layout()
    plt.show()


def draw_volts_cond_profile(failure: Failure, size_obj: float=1, unsigned: bool = False):
    volts_cond_profile = np.array(failure.volts_cond_profile.copy()).transpose() # [edge, time]
    if unsigned: np.abs(volts_cond_profile, out=volts_cond_profile)
    num_edge = volts_cond_profile.shape[0]
    num_broken_edge = len(failure.idxs_edge_broken)

    for idx_time, idx_edge_broken in enumerate(failure.idxs_edge_broken):
        volts_cond_profile[idx_edge_broken, (idx_time + 1):] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=72)

    ax.scatter(0, volts_cond_profile[failure.idxs_edge_broken[0], 0], c="red", s=size_obj, zorder=num_edge)
    for idx_edge, volt_profile in enumerate(volts_cond_profile):
        if idx_edge in failure.idxs_edge_broken:
            ax.plot(range(num_broken_edge), volt_profile, c="red", lw=size_obj, zorder=num_edge)
        else:
            ax.plot(range(num_broken_edge), volt_profile, c="black", lw=size_obj)
    
    plt.tight_layout()
    plt.show()

