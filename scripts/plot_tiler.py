import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

if __name__ == "__main__":
    jname = sys.argv[1]
    with Path(jname).open(mode="r") as fid:
        data = json.load(fid)

    shape = data["shape"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_ylim(0, shape[0])
    ax.set_xlim(0, shape[1])
    sum_diag = 0
    sum_diag2 = 0
    for tile in data["tiles"]:
        p = tile["bounds"]
        ax.add_patch(
            Rectangle(
                (p[1], p[0]),
                p[3] - p[1],
                p[2] - p[0],
                facecolor="none",
                linewidth=1,
                edgecolor="k",
            )
        )

    plt.show()
