import numpy as np
import matplotlib.pyplot as plt

from src.array import Array
from src.failure import Failure


# from src.draw import draw_volts_edge_profile
# draw_volts_edge_profile(array, failure, size_obj=0.1)
def draw_volts_edge_profile(array: Array, failure: Failure, size_obj: float=1, end: bool = False):
    volts_edge_profile = np.array(failure.volts_edge_profile.copy()).transpose() # [edge, time]
    if end:
        volts_edge_profile = np.hstack((volts_edge_profile, np.zeros((array.num_edge, 1)))) # adding the time for macroscopic failure
    num_broken_edge = len(failure.idxs_edge_broken)

    for idx_time, idx_edge_broken in enumerate(failure.idxs_edge_broken):
        volts_edge_profile[idx_edge_broken, (idx_time + 1):] = None

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=72)

    ax.scatter(0, volts_edge_profile[failure.idxs_edge_broken[0], 0], c="red", s=size_obj, zorder=array.num_edge)
    for idx_edge, volt_profile in enumerate(volts_edge_profile):
        if idx_edge in failure.idxs_edge_broken:
            ax.plot(range(num_broken_edge + int(end)), volt_profile, c="red", lw=size_obj, zorder=array.num_edge)
        else:
            ax.plot(range(num_broken_edge + int(end)), volt_profile, c="black", lw=size_obj)
            
    plt.tight_layout()
    plt.show()


def draw_volts_cap_profile(array: Array, failure: Failure, size_obj: float=1):
    volts_cap_profile = np.array(failure.volts_cap_profile.copy()).transpose() # [edge, time]
    num_broken_edge = len(failure.idxs_edge_broken)

    for idx_time, idx_edge_broken in enumerate(failure.idxs_edge_broken):
        volts_cap_profile[idx_edge_broken, (idx_time + 1):] = None

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=72)

    ax.scatter(0, volts_cap_profile[failure.idxs_edge_broken[0], 0], c="red", s=size_obj, zorder=array.num_edge)
    for idx_edge, volt_profile in enumerate(volts_cap_profile):
        if idx_edge in failure.idxs_edge_broken:
            ax.plot(range(num_broken_edge), volt_profile, c="red", lw=size_obj, zorder=array.num_edge)
        else:
            ax.plot(range(num_broken_edge), volt_profile, c="black", lw=size_obj)
    
    plt.tight_layout()
    plt.show()