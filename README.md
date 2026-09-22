# ARIS-Nav: Safety-Aware Path Planner for Grid Navigation

**ARIS-Nav** is a lightweight, decoupled path planning engine designed for drone-style grid navigation. It significantly improves upon baseline A* by incorporating distance-transform obstacle clearance fields and non-linear risk costs directly into heuristic search.

---

## Architecture & Integration Contract

ARIS-Nav is strictly **PLANNER-ONLY**. Rendering, physics, perception, and environment playback live in external simulation modules (e.g. Gazebo / AirSim).

All communication with the planning engine flows through **one locked API contract** in `src/planner.py`.

```
                    +------------------------------------+
                    |  External Simulator / Teammate      |
                    |  (Gazebo / AirSim / Custom Player) |
                    +-----------------+------------------+
                                      |
                      occupancy array |  list of (row, col)
                      changed cells   |  waypoints
                                      v
                    +------------------------------------+
                    |         src/planner.py             |
                    |    (Locked Integration Contract)   |
                    +--------+------------------+--------+
                             |                  |
                    initial  |         replan   |
                    plan     v                  v
                       +-----------+      +-------------------+
                       | src/astar |      | src/aris_planner  |
                       +-----------+      +---------+---------+
                                                    |
                                                    v
                                          +-------------------+
                                          |   src/costmap     |
                                          | (EDT & Risk Field)|
                                          +-------------------+
```

### The Locked API (`src/planner.py`)

```python
import numpy as np
import src.planner as planner

# 1. Initial Plan (mode="aris" or mode="astar")
# Input: 2D numpy array (0 = free, 1 = obstacle), start (r, c), goal (r, c)
# Output: list of (row, col) path tuples, or [] if no valid route
path = planner.plan(occupancy_grid, start=(2, 2), goal=(28, 28), mode="aris")

# 2. Dynamic Replanning
# Input: list of changed cells [((row, col), is_occupied), ...], current agent position
# Output: updated list of (row, col) path tuples from current_pos to goal
updated_path = planner.replan([((15, 15), True)], current_pos=(10, 10))
```

*Data crossing the contract boundary consists exclusively of standard Python tuples, lists, and NumPy arrays.* Matplotlib is strictly confined to developer sanity-checking (`src/debug_viz.py`) and is never imported across the API boundary.

---

## Safety Formulation

### 1. Obstacle Clearance
Calculated using the Euclidean Distance Transform (`scipy.ndimage.distance_transform_edt`) on the inverted occupancy mask:
$$\text{clearance}(u) = \text{Euclidean distance to the nearest obstacle (cells)}$$

### 2. Normalized Risk Field
For a configurable safety radius $R_{safe}$ (default $3.0$ cells):
$$\text{risk}(u) = \begin{cases} 0.0 & \text{if } \text{clearance}(u) \ge R_{safe} \\ \frac{R_{safe} - \text{clearance}(u)}{R_{safe}} & \text{if } \text{clearance}(u) < R_{safe} \end{cases}$$
where $\text{risk}(u) \in [0.0, 1.0]$.

### 3. Traversal Cost
Edge transitions to cell $v$ with base Euclidean step cost $c_{base} \in \{1.0, \sqrt{2}\}$:
$$\text{cost}(v, c_{base}) = \begin{cases} \infty & \text{if } v \text{ is occupied} \\ c_{base} \cdot \left(1 + w_r \cdot \text{risk}(v)^p\right) & \text{otherwise} \end{cases}$$
Default hyperparameters: $w_r = 5.0$, $p = 2.0$.

### 4. Octile Distance Heuristic
For 8-connected grid navigation with step costs $1.0$ and $\sqrt{2}$:
$$h(u, v) = \max(\Delta r, \Delta c) + (\sqrt{2} - 1) \min(\Delta r, \Delta c)$$
Because $\text{cost}(v, c_{base}) \ge c_{base}$, the octile heuristic remains strictly admissible.

---

## Benchmark Results

Evaluated on 3 benchmark scenario topologies (35x35 grids) with dynamic obstacle injection mid-trajectory:

| Scenario | Planner | Plan Time (ms) | Replan Time (ms) | Path Length | Min Clearance | Avg Clearance | Nodes Expanded | Collisions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Open Field** | Plain A* | 1.57 ms | 1.05 ms | 42.53 | 1.000 | 3.742 | 285 | 0 |
| **Open Field** | ARIS-Lite | 2.03 ms | 0.91 ms | 47.46 | **2.828** | 4.525 | 332 | 0 |
| **Open Field** | Incremental ARIS | 2.87 ms | **0.63 ms** | 47.46 | **2.828** | **5.231** | **263** | 0 |
| **Narrow Corridor** | Plain A* | 0.27 ms | 0.34 ms | 28.83 | 1.000 | 5.652 | 69 | 0 |
| **Narrow Corridor** | ARIS-Lite | 2.15 ms | 1.44 ms | 36.28 | **1.414** | 6.661 | 337 | 0 |
| **Narrow Corridor** | Incremental ARIS | 2.87 ms | **0.74 ms** | 36.28 | **1.414** | **6.625** | **318** | 0 |
| **Dead-End Maze** | Plain A* | 2.46 ms | 0.89 ms | 52.87 | 1.000 | 2.158 | 524 | 0 |
| **Dead-End Maze** | ARIS-Lite | 4.43 ms | 2.67 ms | 56.38 | **1.414** | 3.753 | 658 | 0 |
| **Dead-End Maze** | Incremental ARIS | 5.61 ms | **0.85 ms** | 56.38 | **1.414** | **4.907** | **560** | 0 |

### Key Takeaways:
1. **Safety Margin**: ARIS increases minimum clearance from $1.000$ (hugging walls/corners) to up to $2.828$ cells ($+182\%$), keeping drones clear of danger zones.
2. **Replanning Speed**: Bounded Incremental ARIS achieves up to **3.1x faster replanning** than planning from scratch by re-using clean downstream paths and restricting search to the perturbed locality.
3. **Robustness**: 0 collisions across all dynamic obstacle injections.

---

## Directory Layout

```
aris-nav/
├── .venv/                         # Virtual environment
├── requirements.txt               # Dependencies: numpy, scipy, matplotlib
├── src/
│   ├── __init__.py
│   ├── grid.py                    # 2D Grid with 8-connected neighbor lookup
│   ├── debug_viz.py               # Throwaway matplotlib debug visualizer
│   ├── astar.py                   # Baseline 8-connected A* with octile heuristic
│   ├── costmap.py                 # Distance transform EDT, risk field, and costmap
│   ├── aris_planner.py            # ARIS-Lite & bounded incremental replanning
│   ├── planner.py                 # THE LOCKED INTEGRATION API (plan, replan)
│   ├── dynamic_env.py             # Mid-run obstacle injection test harness
│   └── benchmark.py               # Benchmark runner across 3 scenarios
├── experiments/
│   └── run_experiments.py         # Batch statistical experiment runner
├── tests/
│   ├── test_grid.py               # Grid unit tests
│   ├── test_astar.py              # A* correctness tests
│   ├── test_costmap.py            # Clearance & risk tests
│   ├── test_checkpoint.py         # ARIS-Lite vs A* clearance gate
│   └── test_planner_api.py        # Contract tests for plan() and replan()
├── results/
│   ├── metrics.csv                # Benchmark metrics log
│   ├── batch_experiments.csv      # Statistical batch run output
│   └── *.png                      # Generated scenario trajectory maps
└── README.md
```

---

## Execution Instructions

### Running Tests
```bash
.venv/Scripts/python -m unittest discover tests
```

### Running the Benchmark Suite
```bash
.venv/Scripts/python src/benchmark.py
```

### Running Batch Experiments (Statistical Summary)
```bash
.venv/Scripts/python experiments/run_experiments.py
```

---

## Clarifications & Future Work

1. **Simplified Locality-Bounded Replanning**:
   The current replanning implementation is a simplified, bounded re-search inspired by the reuse principles of D* Lite / LPA*. It identifies a dirty radius around perturbed obstacles ($\sim 2 \cdot R_{safe}$), updates local clearance, and searches from the agent's current position to reconnect onto clean downstream path segments. It is **not** a formal implementation of D* Lite or LPA* (which maintains exact RHS values and backwards search trees). Implementing formal D* Lite or Field D* Lite is identified as future work.
2. **2D Grid Domain**:
   The current planning engine is implemented in 2D $(row, col)$ coordinates with 8-connectivity. Extending the distance transform and neighbor lookups to 3D voxel grids with 26-connectivity is straightforward future work.
3. **Perception-Free Contract**:
   ARIS-Nav assumes ground-truth occupancy from upstream SLAM/perception modules. Probabilistic occupancy mapping (e.g. log-odds Bayesian updates) can be integrated upstream.
4. **Decoupled Simulation**:
   Rendering and physics are entirely external. Teammates can connect any simulator (ROS 2 / Gazebo / AirSim) by calling `planner.plan()` and `planner.replan()` directly.
