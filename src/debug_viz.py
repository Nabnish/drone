"""Debug visualizer for sanity checking.

THROW-AWAY MODULE: Never imported by planner.py or any simulation interface.
"""

from typing import Optional, Union
import numpy as np


def show(
    grid,
    path: Optional[list[tuple[int, int]]] = None,
    title: str = "Grid Visualization",
    save_path: Optional[str] = None,
    show_plot: bool = False,
):
    """Sanity-check visualizer using matplotlib.pyplot.imshow.

    Args:
        grid: Grid instance or 2D numpy occupancy array (0=free, 1=obstacle).
        path: Optional list of (row, col) coordinates.
        title: Title for the plot.
        save_path: Optional file path to save the plot image.
        show_plot: Whether to invoke plt.show() interactively.
    """
    import matplotlib.pyplot as plt

    if hasattr(grid, "occupancy"):
        occ = grid.occupancy
        start = grid.start
        goal = grid.goal
    else:
        occ = np.array(grid)
        start = path[0] if path else None
        goal = path[-1] if path else None

    fig, ax = plt.subplots(figsize=(6, 6))

    # Display grid: 0 free (white/light), 1 obstacle (dark)
    ax.imshow(occ, cmap="binary", origin="upper", interpolation="nearest")

    # Plot path if provided
    if path and len(path) > 0:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, color="deepskyblue", linewidth=2.5, label="Path", zorder=3)
        ax.scatter(cols, rows, color="cyan", s=15, zorder=4)

    # Plot start and goal
    if start is not None:
        ax.plot(start[1], start[0], "go", markersize=9, label="Start", zorder=5)
    if goal is not None:
        ax.plot(goal[1], goal[0], "r*", markersize=12, label="Goal", zorder=5)

    ax.set_title(title)
    ax.set_xlabel("Col (X)")
    ax.set_ylabel("Row (Y)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", color="lightgray", linestyle=":", linewidth=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)

    if show_plot:
        plt.show()

    plt.close(fig)
