#!/usr/bin/env python3
"""Render the two water-distribution networks (water01.dat, water02.dat) to PDF.

Node coordinates, the reservoir/consumer split, and the arc lists are taken
directly from the AMPL data files.  Reservoirs are drawn as filled squares,
consumers as circles, and each arc as a directed edge i -> j.

Run from tutorials/water-net/:   python3 figures/make_networks.py
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Network 1 -- water01.dat  (coordinates already in metres)
# ---------------------------------------------------------------------------
NET1 = {
    "reservoirs": {
        "NW": (1200, 3600),
        "E": (4000, 2200),
    },
    "consumers": {
        "CC": (2000, 2300),
        "W": (750, 2400),
        "SW": (900, 1200),
        "S": (2000, 1000),
        "SE": (4000, 900),
        "N": (3700, 3500),
    },
    "arcs": [
        ("NW", "W"), ("NW", "CC"), ("NW", "N"),
        ("E", "N"), ("E", "CC"), ("E", "S"), ("E", "SE"),
        ("CC", "W"), ("CC", "SW"), ("CC", "S"), ("CC", "N"),
        ("S", "SE"), ("S", "SW"), ("SW", "W"),
    ],
    "unit": "m",
}

# ---------------------------------------------------------------------------
# Network 2 -- water02.dat  (raw coordinates in km; the .dat later scales x1000)
# ---------------------------------------------------------------------------
NET2 = {
    "reservoirs": {
        "R1": (0.5, 5.75), "R2": (3.25, 6.0), "R3": (1.75, 4.25),
        "R4": (5.5, 4.25), "R5": (4.25, 2.75), "R6": (2.0, 2.0),
        "R7": (4.0, 1.5), "R8": (0.5, 0.5), "R9": (6.75, 0.25),
    },
    "consumers": {
        "C1": (1.0, 6.5), "C2": (2.5, 6.6), "C3": (4.25, 6.25),
        "C4": (6.25, 6.25), "C5": (1.5, 5.5), "C6": (5.25, 5.25),
        "C7": (0.5, 4.0), "C8": (3.0, 4.5), "C9": (6.75, 4.5),
        "C10": (4.25, 3.75), "C11": (1.5, 3.0), "C12": (3.0, 2.5),
        "C13": (6.25, 2.75), "C14": (1.0, 2.0), "C15": (5.0, 2.0),
        "C16": (1.5, 1.0), "C17": (2.75, 1.25), "C18": (6.0, 1.0),
        "C19": (4.0, 0.25),
    },
    "arcs": [
        ("R1", "C1"), ("R1", "C7"),
        ("R2", "C2"), ("R2", "C3"), ("R2", "C8"),
        ("R3", "C7"), ("R3", "C5"), ("R3", "C8"), ("R3", "C11"),
        ("R4", "C6"), ("R4", "C9"), ("R4", "C10"), ("R4", "C13"),
        ("R5", "C10"), ("R5", "C12"), ("R5", "C13"), ("R5", "C15"),
        ("R6", "C12"), ("R6", "C16"), ("R6", "C17"),
        ("R7", "C15"), ("R7", "C17"), ("R7", "C19"),
        ("R8", "C14"), ("R8", "C16"),
        ("R9", "C18"), ("R9", "C19"),
        ("C1", "C2"),
        ("C4", "C3"),
        ("C5", "C7"), ("C5", "C1"),
        ("C6", "C8"), ("C6", "C4"),
        ("C7", "C11"),
        ("C9", "C4"),
        ("C10", "C8"),
        ("C11", "C12"),
        ("C13", "C9"),
        ("C14", "C11"), ("C14", "C16"),
        ("C16", "C19"),
        ("C18", "C13"), ("C18", "C15"),
        ("C19", "C18"),
    ],
    "unit": "km",
}


def draw_network(net, outfile, node_size, font_size, label_offset, figsize):
    coords = {**net["reservoirs"], **net["consumers"]}

    fig, ax = plt.subplots(figsize=figsize)

    # arcs first, so nodes sit on top
    for i, j in net["arcs"]:
        xi, yi = coords[i]
        xj, yj = coords[j]
        arrow = FancyArrowPatch(
            (xi, yi), (xj, yj),
            arrowstyle="-|>", mutation_scale=11,
            shrinkA=node_size, shrinkB=node_size,
            color="0.45", lw=1.0, zorder=1,
        )
        ax.add_patch(arrow)

    # reservoirs
    rx = [c[0] for c in net["reservoirs"].values()]
    ry = [c[1] for c in net["reservoirs"].values()]
    ax.scatter(rx, ry, marker="s", s=node_size ** 2 * 3.0,
               facecolor="#1f6fb2", edgecolor="black", linewidths=1.0,
               zorder=3, label="reservoir")

    # consumers
    cx = [c[0] for c in net["consumers"].values()]
    cy = [c[1] for c in net["consumers"].values()]
    ax.scatter(cx, cy, marker="o", s=node_size ** 2 * 3.0,
               facecolor="#e8eef4", edgecolor="black", linewidths=1.0,
               zorder=3, label="consumer")

    # labels
    for name, (x, y) in coords.items():
        ax.annotate(name, (x, y), (x + label_offset, y + label_offset),
                    fontsize=font_size, zorder=4)

    ax.set_aspect("equal")
    ax.margins(0.08)
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, fontsize=font_size + 1)

    fig.savefig(os.path.join(HERE, outfile), bbox_inches="tight")
    plt.close(fig)
    print("wrote", outfile)


def main():
    draw_network(NET1, "water01.pdf", node_size=6, font_size=10,
                 label_offset=90, figsize=(5.2, 4.6))
    draw_network(NET2, "water02.pdf", node_size=5, font_size=9,
                 label_offset=0.09, figsize=(6.4, 6.0))


if __name__ == "__main__":
    main()
