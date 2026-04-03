"""
Autonomous Urban Scenario Simulator with Algebraic Topology
==========================================================

This script is a chapter-oriented, end-to-end Python implementation aligned
with Sections 1, 2, 3, and 4 of the article draft:

    "Autonomous Urban Scenario Simulator: Absolute-Zero Generation and Topology Checks"

Main design goals of this implementation
----------------------------------------
1. Represent the city as a coupled digital twin with mobility, energy, and
   public-service layers.
2. Generate candidate scenario programs in the spirit of self-generated,
   verifiable urban tasks.
3. Filter candidates using feasibility, difficulty, novelty, and
   topology-aware admissibility.
4. Use algebraic-topology tooling directly in the code, without GUDHI.
5. Export the outputs already separated into the folders that correspond to
   manuscript Sections 5.1, 5.2, and 5.3.

Topological stack used here
---------------------------
The script is intentionally written around the libraries requested for the
chapter's computational implementation:

    - torch
    - ripser
    - persim
    - kmapper / KeplerMapper
    - toponetx

These libraries are used for:
    - persistent homology on cross-domain urban point clouds,
    - Wasserstein comparisons of persistence diagrams,
    - Mapper graphs for accepted scenario banks,
    - simplicial-complex lifting and Hodge-style diagnostics,
    - nonlinear scenario embeddings and memory indexing.

Important restriction
---------------------
The script does NOT use GUDHI.

Output contract
---------------
The script creates one root results directory with the following structure:

    results_root/
        shared/
            figures/
            data/
        5_1_structural_fragility/
            figures/
            tables/
        5_2_controller_benchmark/
            figures/
            tables/
        5_3_drift_and_refresh/
            figures/
            tables/

This output layout makes it easy to take each figure and table directly into
Sections 5.1, 5.2, and 5.3 of the chapter.

Runtime note
------------
The code is fully defined. No placeholders are left in the implementation.
It is designed to run on a stylized city, so it does not require external
datasets to produce chapter-ready artifacts. In a real deployment, the same
interfaces can ingest calibrated municipal data.
"""

from __future__ import annotations

import copy
import html
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# Matplotlib needs a writable configuration directory in some headless setups.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import argparse
import random
import warnings
from pathlib import Path
from typing import Any

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Optional but required-by-design third-party dependencies for the
# algebraic-topology version of the chapter pipeline.
#
# The script checks these packages explicitly and raises a user-friendly
# message if they are missing. GUDHI is intentionally excluded.
# ---------------------------------------------------------------------
MISSING_PACKAGES: List[str] = []

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None
    MISSING_PACKAGES.append("torch")

TorchModuleBase = nn.Module if nn is not None else object

try:
    from ripser import ripser, Rips
except Exception:
    ripser = None
    Rips = None
    MISSING_PACKAGES.append("ripser")

try:
    from persim import plot_diagrams, wasserstein, bottleneck
except Exception:
    plot_diagrams = None
    wasserstein = None
    bottleneck = None
    MISSING_PACKAGES.append("persim")

try:
    import kmapper as km
except Exception:
    km = None
    MISSING_PACKAGES.append("kmapper")

try:
    import toponetx as tnx
    from toponetx.transform.graph_to_simplicial_complex import graph_to_clique_complex
except Exception:
    tnx = None
    graph_to_clique_complex = None
    MISSING_PACKAGES.append("toponetx")


def check_dependencies() -> None:
    """
    Stop execution early with a clear installation message.

    The chapter explicitly asks for a pipeline based on algebraic-topology
    libraries such as KeplerMapper, TopoNetX, Ripser, Persim, and torch.
    Therefore, the script refuses to proceed silently when any of them is
    unavailable.
    """
    if MISSING_PACKAGES:
        missing = ", ".join(sorted(set(MISSING_PACKAGES)))
        raise ImportError(
            "The following packages are required by this script but are not installed: "
            f"{missing}.\n\n"
            "Install them first, for example:\n"
            "pip install numpy pandas matplotlib networkx scikit-learn torch ripser persim kmapper toponetx\n\n"
            "GUDHI is intentionally not used in this implementation."
        )


def set_global_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Reproducibility is one of the explicit architectural requirements of the
    chapter, so the script centralizes random-state control here.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_output_tree(root: str) -> Dict[str, Path]:
    """
    Create the exact directory tree used by the manuscript.

    Each subsection in Section 5 receives its own folder with separate
    `figures` and `tables` directories so that the exported artifacts can be
    inserted into the draft without manual re-organization.
    """
    root_path = Path(root)
    paths = {
        "root": root_path,
        "shared": root_path / "shared",
        "shared_fig": root_path / "shared" / "figures",
        "shared_data": root_path / "shared" / "data",
        "s51": root_path / "5_1_structural_fragility",
        "s51_fig": root_path / "5_1_structural_fragility" / "figures",
        "s51_tab": root_path / "5_1_structural_fragility" / "tables",
        "s52": root_path / "5_2_controller_benchmark",
        "s52_fig": root_path / "5_2_controller_benchmark" / "figures",
        "s52_tab": root_path / "5_2_controller_benchmark" / "tables",
        "s53": root_path / "5_3_drift_and_refresh",
        "s53_fig": root_path / "5_3_drift_and_refresh" / "figures",
        "s53_tab": root_path / "5_3_drift_and_refresh" / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

SECTION_52_FAMILY_ORDER = [
    "ClinicOutage",
    "CorridorIncident",
    "EVChargingShock",
    "HeatwaveLoad",
    "StadiumEvent",
    "StormCascade",
]

CONTROLLER_ORDER = ["Incumbent", "Reactive", "TopologyAware"]


def ordered_controller_family_loss_matrix(
    controller_df: pd.DataFrame,
    family_order: Optional[Sequence[str]] = None,
    controller_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a family-by-controller loss matrix without assuming that every
    family and every controller is present in the evaluated bank.

    The original implementation used `.loc[...]`, which raises `KeyError`
    whenever one accepted family is absent from `controller_df`.
    """
    ordered_families = list(family_order or SECTION_52_FAMILY_ORDER)
    ordered_controllers = list(controller_order or CONTROLLER_ORDER)

    if controller_df is None or controller_df.empty:
        return pd.DataFrame(
            np.nan,
            index=ordered_families,
            columns=ordered_controllers,
            dtype=float,
        )

    family_controller = (
        controller_df.groupby(["family", "controller"])["loss"].mean().unstack()
    )
    family_controller = family_controller.reindex(
        index=ordered_families,
        columns=ordered_controllers,
    )
    return family_controller.astype(float)


def nan_aware_imshow(axis, values: pd.DataFrame, cmap_name: str = "Blues"):
    """
    Render a heatmap that preserves missing combinations as masked cells.
    """
    colormap = copy.copy(plt.get_cmap(cmap_name))
    if hasattr(colormap, "set_bad"):
        colormap.set_bad(color="#f2f2f2")
    masked_values = np.ma.masked_invalid(values.to_numpy(dtype=float))
    return axis.imshow(masked_values, aspect="auto", cmap=colormap)


def annotate_dataframe_cells(axis, values: pd.DataFrame, fontsize: int = 8) -> None:
    """
    Write numeric annotations for a dataframe-backed heatmap.

    Missing values are shown explicitly as `NA` instead of being coerced to
    zero, which would distort the interpretation of the benchmark figure.
    """
    for i_row in range(values.shape[0]):
        for j_col in range(values.shape[1]):
            cell_value = values.iloc[i_row, j_col]
            label = "NA" if pd.isna(cell_value) else f"{float(cell_value):.2f}"
            axis.text(j_col, i_row, label, ha="center", va="center", fontsize=fontsize)


def safe_controller_win_rate(
    pivot: pd.DataFrame,
    controller_name: str,
    incumbent_name: str = "Incumbent",
) -> float:
    """
    Compute a win rate only on rows where both controllers are available.
    """
    if pivot is None or pivot.empty:
        return float("nan")
    if controller_name not in pivot.columns or incumbent_name not in pivot.columns:
        return float("nan")

    comparable = pivot[[controller_name, incumbent_name]].dropna()
    if comparable.empty:
        return float("nan")
    return float((comparable[controller_name] < comparable[incumbent_name]).mean())


def finite_diagram(diagram: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """
    Keep only finite birth-death pairs.

    Ripser may return infinite deaths for some H0 components. Persim's distance
    functions expect finite coordinates when diagrams are compared directly, so
    the script filters them here.
    """
    array = np.asarray(diagram, dtype=float)
    if array.size == 0:
        return np.zeros((0, 2), dtype=float)
    array = array.reshape(-1, 2)
    mask = np.isfinite(array[:, 0]) & np.isfinite(array[:, 1])
    return array[mask]


def diagram_with_fallback(diagram: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """
    Ensure that a persistence diagram passed to Persim is never empty.

    When a diagram is empty, a single diagonal point is injected so that
    Wasserstein and bottleneck distances remain numerically well defined.
    """
    finite = finite_diagram(diagram)
    if finite.size == 0:
        return np.array([[0.0, 0.0]], dtype=float)
    return finite


def persistent_entropy(diagram: np.ndarray | Sequence[Sequence[float]]) -> float:
    """
    Compute persistent entropy from finite lifetimes.

    The entropy is used as a compact summary of how concentrated or diffuse the
    diagram mass is. It helps distinguish a clean structural regime from a noisy
    proliferation of short-lived topological features.
    """
    finite = finite_diagram(diagram)
    if finite.size == 0:
        return 0.0

    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 1e-12)]
    if lifetimes.size == 0:
        return 0.0

    probs = lifetimes / np.sum(lifetimes)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def sparse_to_dense(matrix: Any) -> np.ndarray:
    """
    Convert sparse or array-like matrices to a dense NumPy array.

    TopoNetX incidence matrices can be sparse. This helper makes the later
    linear-algebra code agnostic to the exact matrix container.
    """
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


class ScenarioEmbeddingAutoencoder(TorchModuleBase):
    """
    Small nonlinear autoencoder used to build a latent memory of scenario-bank
    fingerprints.

    The latent representation is not treated as a black-box prediction target.
    Instead, it acts as a compact indexing space for:
    - scenario-bank visualization,
    - drift-aware comparison,
    - coverage diagnostics across operating epochs.
    """

    def __init__(self, input_dim: int, latent_dim: int = 4) -> None:
        super().__init__()
        hidden_1 = max(24, input_dim * 2)
        hidden_2 = max(12, input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, input_dim),
        )

    def encode(self, x_tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_tensor)

    def forward(self, x_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_tensor = self.encoder(x_tensor)
        reconstruction = self.decoder(z_tensor)
        return reconstruction, z_tensor



@dataclass
class ScenarioProgram:
    """
    Compact representation of a scenario program.

    Each program is a typed perturbation sequence with:
    - a family label,
    - an intensity,
    - an activation window,
    - targeted mobility edges,
    - targeted energy nodes,
    - an optional targeted facility.
    """

    family: str
    intensity: float
    start: int
    duration: int
    target_edges: List[Tuple[int, int]] = field(default_factory=list)
    target_energy_nodes: List[int] = field(default_factory=list)
    target_facility: Optional[int] = None
    description: str = ""
    seed: int = 0


class AutonomousUrbanScenarioSimulator:
    """
    Self-contained research simulator used to generate the empirical material
    for Sections 5.1, 5.2, and 5.3 of the chapter.

    The model is intentionally stylized:
    it is compact enough to run quickly,
    but it preserves the full chapter logic:
    data -> calibration -> coupled twin -> scenario generation ->
    topology checks -> scenario bank -> benchmarking -> drift refresh.
    """

    def __init__(self, master_seed: int = 2026) -> None:
        self.master_seed = master_seed
        self.rng = np.random.default_rng(master_seed)

        # Public names used across tables and plots.
        self.families = [
            "CorridorIncident",
            "HeatwaveLoad",
            "ClinicOutage",
            "StadiumEvent",
            "StormCascade",
            "EVChargingShock",
        ]
        self.family_to_idx = {name: idx for idx, name in enumerate(self.families)}

        # Build the original city, store a clean snapshot, and compute baseline.
        self._build_city()
        self.original_state = self._snapshot_state()
        self._refresh_baseline()

    # ------------------------------------------------------------------
    # Core city construction
    # ------------------------------------------------------------------
    def _build_city(self) -> None:
        """
        Build the stylized smart-city environment.

        The city contains:
        - 12 mobility zones in a grid-like urban layout,
        - 8 energy nodes in a feeder-tree structure,
        - 4 public-service facilities,
        - baseline profiles for demand, load, and service arrivals.
        """
        # --- Zone coordinates -------------------------------------------------
        self.coords: Dict[int, Tuple[int, int]] = {}
        idx = 0
        for row in range(3):
            for col in range(4):
                self.coords[idx] = (col, 2 - row)
                idx += 1

        # --- Mobility graph --------------------------------------------------
        self.mobility_graph = nx.Graph()
        for node, (x_coord, y_coord) in self.coords.items():
            self.mobility_graph.add_node(node, pos=(x_coord, y_coord))

        base_edges: List[Tuple[int, int]] = []
        for node_i, (x_i, y_i) in self.coords.items():
            for node_j, (x_j, y_j) in self.coords.items():
                if node_i < node_j and abs(x_i - x_j) + abs(y_i - y_j) == 1:
                    base_edges.append((node_i, node_j))

        # Add cross-links and express links to create route alternatives.
        base_edges += [
            (0, 5),
            (1, 6),
            (2, 7),
            (4, 9),
            (5, 10),
            (6, 11),
            (1, 4),
            (2, 5),
            (6, 9),
            (7, 10),
        ]
        base_edges = list(dict.fromkeys(tuple(sorted(edge)) for edge in base_edges))

        local_rng = np.random.default_rng(42)
        for u_node, v_node in base_edges:
            x_u, y_u = self.coords[u_node]
            x_v, y_v = self.coords[v_node]
            distance = float(((x_u - x_v) ** 2 + (y_u - y_v) ** 2) ** 0.5)

            free_flow_time = float(4 + 3 * distance + local_rng.normal(0, 0.3))
            capacity = float(local_rng.integers(220, 420))

            if distance > 1.1:
                capacity += 80
                free_flow_time += 1.0

            self.mobility_graph.add_edge(
                u_node,
                v_node,
                distance=distance,
                t0=max(3.5, free_flow_time),
                capacity=capacity,
            )

        # --- Zone populations and jobs --------------------------------------
        self.populations = np.array(
            [1400, 1900, 1800, 1500, 1700, 2600, 2500, 1900, 1300, 2100, 2000, 1600],
            dtype=float,
        )
        self.jobs = np.array(
            [1500, 2100, 2300, 1700, 1300, 2800, 2900, 1600, 1200, 2000, 2100, 1800],
            dtype=float,
        )

        # --- Public-service layer -------------------------------------------
        self.facilities = [1, 5, 6, 10]
        self.facility_service_rates = {1: 250.0, 5: 320.0, 6: 310.0, 10: 270.0}

        # --- Energy graph ----------------------------------------------------
        self.energy_nodes = list(range(8))
        self.zone_to_energy_node = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 1,
            5: 2,
            6: 4,
            7: 5,
            8: 4,
            9: 6,
            10: 6,
            11: 7,
        }

        self.energy_graph = nx.Graph()
        for node in self.energy_nodes:
            self.energy_graph.add_node(node)

        tree_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (4, 6), (5, 7)]
        for u_node, v_node in tree_edges:
            limit = float(local_rng.integers(360, 560) * 1.20)
            resistance = float(local_rng.uniform(0.008, 0.025))
            self.energy_graph.add_edge(u_node, v_node, limit=limit, r=resistance)

        self.base_load = {
            0: 110.0,
            1: 120.0,
            2: 105.0,
            3: 65.0,
            4: 85.0,
            5: 75.0,
            6: 115.0,
            7: 80.0,
        }
        self.distributed_generation = {
            0: 0.0,
            1: 25.0,
            2: 20.0,
            3: 15.0,
            4: 10.0,
            5: 18.0,
            6: 20.0,
            7: 12.0,
        }

        # --- Time horizon and exogenous profiles ----------------------------
        self.horizon = 8
        self.demand_profile = np.array([0.75, 0.95, 1.20, 1.35, 1.15, 0.95, 0.85, 0.70])
        self.load_profile = np.array([0.82, 0.90, 1.00, 1.08, 1.18, 1.22, 1.10, 0.90])
        self.service_profile = np.array([0.88, 0.96, 1.04, 1.10, 1.08, 1.00, 0.94, 0.86])

        # --- Distance matrix and OD baseline -------------------------------
        self.distance_matrix = np.zeros((12, 12), dtype=float)
        for i_idx in range(12):
            for j_idx in range(12):
                if i_idx == j_idx:
                    self.distance_matrix[i_idx, j_idx] = 0.5
                else:
                    x_i, y_i = self.coords[i_idx]
                    x_j, y_j = self.coords[j_idx]
                    self.distance_matrix[i_idx, j_idx] = (
                        ((x_i - x_j) ** 2 + (y_i - y_j) ** 2) ** 0.5 + 0.6
                    )

        gravity = np.outer(self.populations, self.jobs) / (self.distance_matrix ** 1.35)
        np.fill_diagonal(gravity, 0.0)
        gravity = gravity / gravity.sum() * 3200.0
        self.gravity = gravity

        # Critical mobility edges are useful both for scenario sampling and
        # for topology-aware policy learning.
        betweenness = nx.edge_betweenness_centrality(self.mobility_graph, weight="t0")
        self.critical_edges = sorted(
            self.mobility_graph.edges(), key=lambda edge: betweenness[edge], reverse=True
        )[:10]

    def _snapshot_state(self) -> Dict[str, object]:
        """
        Create a deep snapshot of the entire city state.

        This snapshot is used when the script evaluates structural drift.
        """
        return {
            "mobility_graph": copy.deepcopy(self.mobility_graph),
            "energy_graph": copy.deepcopy(self.energy_graph),
            "gravity": self.gravity.copy(),
            "facility_service_rates": copy.deepcopy(self.facility_service_rates),
            "base_load": copy.deepcopy(self.base_load),
            "populations": self.populations.copy(),
            "jobs": self.jobs.copy(),
            "demand_profile": self.demand_profile.copy(),
            "load_profile": self.load_profile.copy(),
            "service_profile": self.service_profile.copy(),
        }

    def restore_original_state(self) -> None:
        """
        Restore the simulator to the pristine pre-drift state.
        """
        state = copy.deepcopy(self.original_state)
        self.mobility_graph = state["mobility_graph"]
        self.energy_graph = state["energy_graph"]
        self.gravity = state["gravity"]
        self.facility_service_rates = state["facility_service_rates"]
        self.base_load = state["base_load"]
        self.populations = state["populations"]
        self.jobs = state["jobs"]
        self.demand_profile = state["demand_profile"]
        self.load_profile = state["load_profile"]
        self.service_profile = state["service_profile"]
        self._refresh_baseline()

    def _refresh_baseline(self) -> None:
        """
        Recompute the baseline summary after any structural modification.

        This is essential because:
        - travel ratios,
        - service ratios,
        - topology similarity scores,
        all depend on the current baseline city.
        """
        baseline_timeseries, peak_index = self.simulate_scenario(
            program=None,
            controller_name="Incumbent",
            learned_assets=None,
        )
        self.baseline_summary = {
            "mobility_mean_tt": float(np.mean([row["mobility_mean_tt"] for row in baseline_timeseries])),
            "mean_wait": float(np.mean([row["mean_wait"] for row in baseline_timeseries])),
            "access_dispersion": float(
                np.mean([row["access_dispersion"] for row in baseline_timeseries])
            ),
            "topology": baseline_timeseries[peak_index]["topology"],
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    @staticmethod
    def edge_key(edge: Tuple[int, int]) -> Tuple[int, int]:
        """
        Use an ordered tuple as a stable edge key.
        """
        u_node, v_node = edge
        return tuple(sorted((u_node, v_node)))

    def _route_assignment(
        self, edge_t0: Dict[Tuple[int, int], float], edge_cap: Dict[Tuple[int, int], float], od_matrix: np.ndarray
    ) -> Tuple[Dict[Tuple[int, int], float], np.ndarray]:
        """
        Assign OD demand to shortest paths.

        The route choice is a deliberately simple all-or-nothing assignment:
        it is fast, transparent, and sufficient for the chapter illustration.
        """
        assignment_graph = nx.Graph()
        assignment_graph.add_nodes_from(self.mobility_graph.nodes(data=True))

        for u_node, v_node in self.mobility_graph.edges():
            key = self.edge_key((u_node, v_node))
            base_time = edge_t0[key]
            capacity_factor = max(edge_cap[key] / self.mobility_graph[u_node][v_node]["capacity"], 0.08)
            route_weight = base_time / math.sqrt(capacity_factor)
            assignment_graph.add_edge(u_node, v_node, weight=route_weight)

        edge_flow: Dict[Tuple[int, int], float] = defaultdict(float)
        od_shortest_path_t0 = np.zeros((12, 12), dtype=float)

        for origin in range(12):
            for destination in range(12):
                if origin == destination:
                    continue
                demand_value = float(od_matrix[origin, destination])
                if demand_value <= 0:
                    continue

                path = nx.shortest_path(
                    assignment_graph,
                    source=origin,
                    target=destination,
                    weight="weight",
                )

                for node_a, node_b in zip(path[:-1], path[1:]):
                    edge_flow[self.edge_key((node_a, node_b))] += demand_value

                od_shortest_path_t0[origin, destination] = sum(
                    edge_t0[self.edge_key((node_a, node_b))]
                    for node_a, node_b in zip(path[:-1], path[1:])
                )

        return edge_flow, od_shortest_path_t0

    def _compute_bpr_times(
        self, edge_flow: Dict[Tuple[int, int], float], edge_cap: Dict[Tuple[int, int], float]
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        """
        Compute travel times with a BPR-style congestion function.

        The parameters are selected so that:
        - the baseline city remains stable,
        - severe corridor stress still produces meaningful burden escalation.
        """
        edge_time: Dict[Tuple[int, int], float] = {}
        residual_capacity: Dict[Tuple[int, int], float] = {}

        for u_node, v_node, data in self.mobility_graph.edges(data=True):
            key = self.edge_key((u_node, v_node))
            free_flow_time = float(data["t0"])
            effective_cap = max(edge_cap[key], 1e-3)
            flow_value = float(edge_flow.get(key, 0.0))

            alpha = 0.28
            beta = 4.5
            utilization = max(flow_value / effective_cap, 0.0)

            travel_time = free_flow_time * (1 + alpha * utilization ** beta)
            if utilization > 1:
                travel_time += free_flow_time * 1.4 * (utilization - 1) ** 2

            edge_time[key] = float(travel_time)
            residual_capacity[key] = max(0.0, effective_cap - flow_value) / effective_cap

        return edge_time, residual_capacity

    def _all_pairs_travel(self, edge_time: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        Build all-pairs shortest travel times from the current edge times.
        """
        weighted_graph = nx.Graph()
        weighted_graph.add_nodes_from(self.mobility_graph.nodes(data=True))

        for u_node, v_node in self.mobility_graph.edges():
            weighted_graph.add_edge(
                u_node,
                v_node,
                weight=edge_time[self.edge_key((u_node, v_node))],
            )

        travel_matrix = np.zeros((12, 12), dtype=float)
        for source_node, distances in nx.all_pairs_dijkstra_path_length(weighted_graph, weight="weight"):
            for destination_node, distance in distances.items():
                travel_matrix[source_node, destination_node] = float(distance)

        return travel_matrix

    # ------------------------------------------------------------------
    # Scenario generation and controller logic
    # ------------------------------------------------------------------
    def sample_scenario(self, seed: int, family: Optional[str] = None) -> ScenarioProgram:
        """
        Sample a new scenario program.

        Families are intentionally heterogeneous to reflect the book's emphasis
        on cross-domain computational intelligence for smart cities.
        """
        local_rng = np.random.default_rng(seed)
        scenario_family = family or str(local_rng.choice(self.families))
        intensity = float(local_rng.uniform(0.25, 0.95))
        start = int(local_rng.integers(1, 5))
        duration = int(local_rng.integers(2, 4))

        target_edges: List[Tuple[int, int]] = []
        target_energy_nodes: List[int] = []
        target_facility: Optional[int] = None
        description = ""

        if scenario_family == "CorridorIncident":
            selected = local_rng.choice(np.array(self.critical_edges, dtype=object), size=local_rng.integers(2, 4), replace=False)
            target_edges = [self.edge_key(tuple(edge)) for edge in selected]
            description = "Capacity loss on critical corridors."
        elif scenario_family == "HeatwaveLoad":
            target_energy_nodes = list(map(int, local_rng.choice(self.energy_nodes[1:], size=local_rng.integers(2, 4), replace=False)))
            description = "High temperature increases electrical load and service demand."
        elif scenario_family == "ClinicOutage":
            target_facility = int(local_rng.choice(self.facilities))
            selected = local_rng.choice(np.array(list(self.mobility_graph.edges()), dtype=object), size=2, replace=False)
            target_edges = [self.edge_key(tuple(edge)) for edge in selected]
            description = "Temporary service outage combined with local detours."
        elif scenario_family == "StadiumEvent":
            selected = local_rng.choice(np.array(self.critical_edges[:6], dtype=object), size=2, replace=False)
            target_edges = [self.edge_key(tuple(edge)) for edge in selected]
            description = "Event-related demand surge concentrated in central districts."
        elif scenario_family == "StormCascade":
            selected_edges = local_rng.choice(np.array(self.critical_edges[:8], dtype=object), size=local_rng.integers(3, 5), replace=False)
            target_edges = [self.edge_key(tuple(edge)) for edge in selected_edges]
            target_energy_nodes = list(map(int, local_rng.choice(self.energy_nodes[1:], size=local_rng.integers(2, 4), replace=False)))
            target_facility = int(local_rng.choice(self.facilities))
            description = "Coupled shock on transport, grid, and services."
        elif scenario_family == "EVChargingShock":
            selected_edges = local_rng.choice(np.array(self.critical_edges[:8], dtype=object), size=2, replace=False)
            target_edges = [self.edge_key(tuple(edge)) for edge in selected_edges]
            target_energy_nodes = list(map(int, local_rng.choice([2, 4, 5, 6], size=local_rng.integers(2, 3), replace=False)))
            description = "Charging surge stresses feeder branches and mobility access."

        return ScenarioProgram(
            family=scenario_family,
            intensity=intensity,
            start=start,
            duration=duration,
            target_edges=target_edges,
            target_energy_nodes=target_energy_nodes,
            target_facility=target_facility,
            description=description,
            seed=seed,
        )

    def mutate_scenario(self, program: ScenarioProgram, seed: int) -> ScenarioProgram:
        """
        Create a mutated scenario program from a parent candidate.

        This lightweight mutation operator approximates the evolutionary loop
        described in the chapter.
        """
        local_rng = np.random.default_rng(seed)

        mutated = ScenarioProgram(
            family=program.family if local_rng.random() > 0.15 else str(local_rng.choice(self.families)),
            intensity=float(np.clip(program.intensity + local_rng.normal(0.0, 0.12), 0.20, 0.98)),
            start=int(np.clip(program.start + local_rng.integers(-1, 2), 1, 4)),
            duration=int(np.clip(program.duration + local_rng.integers(-1, 2), 2, 4)),
            target_edges=list(program.target_edges),
            target_energy_nodes=list(program.target_energy_nodes),
            target_facility=program.target_facility,
            description=program.description,
            seed=seed,
        )

        # With some probability, or if the family changed, resample the targets.
        if local_rng.random() < 0.35 or mutated.family != program.family:
            refreshed = self.sample_scenario(seed=seed, family=mutated.family)
            mutated.target_edges = refreshed.target_edges
            mutated.target_energy_nodes = refreshed.target_energy_nodes
            mutated.target_facility = refreshed.target_facility
            mutated.description = refreshed.description

        return mutated

    def scenario_state(self, program: Optional[ScenarioProgram], time_index: int) -> Dict[str, object]:
        """
        Convert a scenario program into time-local modifiers.

        The output is intentionally explicit because:
        - it mirrors the chapter notation,
        - it keeps the simulation logic readable,
        - it makes scenario provenance auditable.
        """
        active = (
            program is not None
            and program.start <= time_index < program.start + program.duration
        )

        modifiers = {
            "demand_mult": 1.0,
            "service_demand_mult": 1.0,
            "edge_cap_factor": defaultdict(lambda: 1.0),
            "energy_load_mult": defaultdict(lambda: 1.0),
            "dg_loss_factor": defaultdict(lambda: 0.0),
            "facility_mu_factor": defaultdict(lambda: 1.0),
            "zone_bias": np.ones(12, dtype=float),
        }

        if not active or program is None:
            return modifiers

        intensity = program.intensity
        family = program.family

        if family == "CorridorIncident":
            modifiers["demand_mult"] = 1.08 + 0.22 * intensity
            for edge in program.target_edges:
                modifiers["edge_cap_factor"][edge] = max(0.10, 1.0 - (0.45 + 0.60 * intensity))

        elif family == "HeatwaveLoad":
            modifiers["demand_mult"] = 1.03 + 0.05 * intensity
            modifiers["service_demand_mult"] = 1.12 + 0.30 * intensity
            for node in program.target_energy_nodes:
                modifiers["energy_load_mult"][node] = 1.0 + 0.40 + 0.70 * intensity
                modifiers["dg_loss_factor"][node] = 0.15 + 0.30 * intensity

        elif family == "ClinicOutage":
            modifiers["service_demand_mult"] = 1.08 + 0.15 * intensity
            if program.target_facility is not None:
                modifiers["facility_mu_factor"][program.target_facility] = max(
                    0.08, 1.0 - (0.55 + 0.40 * intensity)
                )
            for edge in program.target_edges:
                modifiers["edge_cap_factor"][edge] = max(0.20, 1.0 - (0.25 + 0.35 * intensity))

        elif family == "StadiumEvent":
            modifiers["demand_mult"] = 1.14 + 0.40 * intensity
            modifiers["service_demand_mult"] = 1.05 + 0.10 * intensity
            bias = np.array([0.75, 1.00, 1.35, 1.45, 0.80, 1.05, 1.40, 1.55, 0.70, 0.95, 1.25, 1.35], dtype=float)
            modifiers["zone_bias"] = 1 + (bias - 1) * intensity
            for edge in program.target_edges:
                modifiers["edge_cap_factor"][edge] = max(0.28, 1.0 - (0.18 + 0.30 * intensity))

        elif family == "StormCascade":
            modifiers["demand_mult"] = 1.10 + 0.22 * intensity
            modifiers["service_demand_mult"] = 1.14 + 0.26 * intensity
            for edge in program.target_edges:
                modifiers["edge_cap_factor"][edge] = max(0.08, 1.0 - (0.45 + 0.55 * intensity))
            for node in program.target_energy_nodes:
                modifiers["energy_load_mult"][node] = 1.0 + 0.28 + 0.52 * intensity
                modifiers["dg_loss_factor"][node] = 0.25 + 0.38 * intensity
            if program.target_facility is not None:
                modifiers["facility_mu_factor"][program.target_facility] = max(
                    0.12, 1.0 - (0.35 + 0.40 * intensity)
                )

        elif family == "EVChargingShock":
            modifiers["demand_mult"] = 1.05 + 0.12 * intensity
            modifiers["service_demand_mult"] = 1.02 + 0.05 * intensity
            for node in program.target_energy_nodes:
                modifiers["energy_load_mult"][node] = 1.0 + 0.35 + 0.70 * intensity
            for edge in program.target_edges:
                modifiers["edge_cap_factor"][edge] = max(0.30, 1.0 - (0.12 + 0.25 * intensity))

        return modifiers

    def edge_capacity_restoration(
        self,
        controller_name: str,
        scenario: Optional[ScenarioProgram],
        time_index: int,
        learned_assets: Optional[Dict[str, object]] = None,
    ) -> Dict[Tuple[int, int], float]:
        """
        Mobility-side control action.

        The policies are intentionally differentiated:
        - Incumbent applies weak generic response,
        - Reactive responds to explicitly targeted assets,
        - TopologyAware combines scenario-local reaction with learned critical assets.
        """
        restore = defaultdict(lambda: 1.0)
        active = (
            scenario is not None
            and scenario.start <= time_index < scenario.start + scenario.duration
        )
        if not active:
            return restore

        if controller_name == "Incumbent":
            for edge in scenario.target_edges[:1]:
                restore[edge] = 1.06

        elif controller_name == "Reactive":
            for edge in scenario.target_edges[:2]:
                restore[edge] = 1.18 + 0.05 * scenario.intensity

        elif controller_name == "TopologyAware":
            learned_assets = learned_assets or {}
            targets = list(scenario.target_edges[:2]) + list(learned_assets.get("critical_edges", [])[:2])
            for edge in set(map(self.edge_key, targets)):
                restore[edge] = 1.24 + 0.07 * scenario.intensity

        return restore

    def energy_response(
        self,
        controller_name: str,
        scenario: Optional[ScenarioProgram],
        time_index: int,
        learned_assets: Optional[Dict[str, object]] = None,
    ) -> Dict[int, float]:
        """
        Energy-side demand response action.
        """
        shed = defaultdict(lambda: 0.0)
        active = (
            scenario is not None
            and scenario.start <= time_index < scenario.start + scenario.duration
        )
        if not active:
            return shed

        if controller_name == "Incumbent":
            for node in scenario.target_energy_nodes[:1]:
                shed[node] = 0.04

        elif controller_name == "Reactive":
            for node in scenario.target_energy_nodes[:2]:
                shed[node] = 0.12 + 0.06 * scenario.intensity

        elif controller_name == "TopologyAware":
            learned_assets = learned_assets or {}
            targets = list(scenario.target_energy_nodes[:2]) + list(
                learned_assets.get("critical_energy_nodes", [])[:2]
            )
            for node in set(targets):
                shed[node] = 0.18 + 0.08 * scenario.intensity

        return shed

    def service_response(
        self,
        controller_name: str,
        scenario: Optional[ScenarioProgram],
        time_index: int,
        learned_assets: Optional[Dict[str, object]] = None,
    ) -> Dict[int, float]:
        """
        Public-service staffing response.

        This method intentionally allows the topology-aware controller to
        strengthen both the directly affected facility and recurrently critical
        facilities learned from the scenario bank.
        """
        boost = defaultdict(lambda: 1.0)
        active = (
            scenario is not None
            and scenario.start <= time_index < scenario.start + scenario.duration
        )
        if not active:
            return boost

        if controller_name == "Incumbent":
            if scenario.target_facility is not None:
                boost[scenario.target_facility] = 1.04

        elif controller_name == "Reactive":
            if scenario.target_facility is not None:
                boost[scenario.target_facility] = 1.18 + 0.06 * scenario.intensity
                alternatives = [facility for facility in self.facilities if facility != scenario.target_facility]
                if alternatives:
                    nearest_alternative = min(alternatives, key=lambda facility: abs(facility - scenario.target_facility))
                    boost[nearest_alternative] = 1.08 + 0.03 * scenario.intensity
            else:
                boost[self.facilities[0]] = 1.05

        elif controller_name == "TopologyAware":
            learned_assets = learned_assets or {}
            targets = [scenario.target_facility] if scenario.target_facility is not None else []
            targets += list(learned_assets.get("critical_facilities", [])[:2])
            for facility in {facility for facility in targets if facility is not None}:
                boost[facility] = 1.22 + 0.08 * scenario.intensity

        return boost

    # ------------------------------------------------------------------
    # Domain simulation blocks
    # ------------------------------------------------------------------
    def service_step(
        self,
        travel_matrix: np.ndarray,
        modifiers: Dict[str, object],
        controller_name: str,
        scenario: Optional[ScenarioProgram],
        time_index: int,
        learned_assets: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """
        Simulate service assignments and queue burden.

        Each zone is assigned to the facility that minimizes generalized access
        cost, which includes travel time and queueing burden.
        """
        demand_zone = (
            self.populations / self.populations.sum()
        ) * (520.0 * self.service_profile[time_index] * float(modifiers["service_demand_mult"]))
        demand_zone = demand_zone * modifiers["zone_bias"]

        service_boost = self.service_response(
            controller_name=controller_name,
            scenario=scenario,
            time_index=time_index,
            learned_assets=learned_assets,
        )

        effective_service_rates: Dict[int, float] = {}
        for facility in self.facilities:
            effective_service_rates[facility] = (
                self.facility_service_rates[facility]
                * float(modifiers["facility_mu_factor"][facility])
                * float(service_boost[facility])
            )

        # First pass: travel-only assignment
        arrival_first_pass: Dict[int, float] = defaultdict(float)
        for zone in range(12):
            travel_costs = {facility: travel_matrix[zone, facility] for facility in self.facilities}
            best_facility = min(travel_costs, key=travel_costs.get)
            arrival_first_pass[best_facility] += float(demand_zone[zone])

        preliminary_waits: Dict[int, float] = {}
        for facility in self.facilities:
            lambda_value = float(arrival_first_pass[facility])
            mu_value = float(effective_service_rates[facility])
            slack = max(mu_value - lambda_value, 1.0)
            preliminary_waits[facility] = min(lambda_value / slack, 25.0)

        # Second pass: generalized cost assignment
        arrival_second_pass: Dict[int, float] = defaultdict(float)
        access_costs = np.zeros(12, dtype=float)

        for zone in range(12):
            generalized_costs = {
                facility: travel_matrix[zone, facility] + 0.6 * preliminary_waits[facility]
                for facility in self.facilities
            }
            best_facility = min(generalized_costs, key=generalized_costs.get)
            access_costs[zone] = float(generalized_costs[best_facility])
            arrival_second_pass[best_facility] += float(demand_zone[zone])

        final_waits: Dict[int, float] = {}
        for facility in self.facilities:
            lambda_value = float(arrival_second_pass[facility])
            mu_value = float(effective_service_rates[facility])
            slack = max(mu_value - lambda_value, 1.0)
            final_waits[facility] = min(lambda_value / slack, 25.0)

        return {
            "facility_arrival": dict(arrival_second_pass),
            "facility_waits": final_waits,
            "mean_wait": float(np.mean(list(final_waits.values()))),
            "access_costs": access_costs,
            "access_dispersion": float(np.std(access_costs) / (np.mean(access_costs) + 1e-6)),
            "reachability_ratio": float(np.mean(access_costs <= 24.0)),
        }

    def energy_step(
        self,
        modifiers: Dict[str, object],
        controller_name: str,
        scenario: Optional[ScenarioProgram],
        time_index: int,
        demand_total: float,
        learned_assets: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """
        Simulate feeder loading and overload pressure.

        The electric model is deliberately simplified but preserves the main
        chapter insight: scenario stress propagates through infrastructure
        coupling rather than remaining isolated in one subsystem.
        """
        load_shed = self.energy_response(
            controller_name=controller_name,
            scenario=scenario,
            time_index=time_index,
            learned_assets=learned_assets,
        )

        node_load: Dict[int, float] = {}
        ev_load_total = 0.06 * demand_total
        ev_weights = {0: 0.02, 1: 0.08, 2: 0.18, 3: 0.05, 4: 0.18, 5: 0.17, 6: 0.22, 7: 0.10}

        for node in self.energy_nodes:
            load_multiplier = float(modifiers["energy_load_mult"][node])
            dg_loss = float(modifiers["dg_loss_factor"][node])

            load_value = self.base_load[node] * self.load_profile[time_index] * load_multiplier
            load_value += ev_load_total * ev_weights[node]
            load_value *= (1.0 - float(load_shed[node]))

            net_load = max(load_value - self.distributed_generation[node] * (1.0 - dg_loss), 0.0)
            node_load[node] = float(net_load)

        parent = {0: None}
        bfs_order = list(nx.bfs_tree(self.energy_graph, source=0))
        for u_node, v_node in nx.bfs_edges(self.energy_graph, source=0):
            parent[v_node] = u_node

        subtree_sum = {node: node_load[node] for node in self.energy_nodes}
        for node in reversed(bfs_order[1:]):
            subtree_sum[parent[node]] += subtree_sum[node]

        branch_flow: Dict[Tuple[int, int], float] = {}
        branch_excess: List[float] = []

        for u_node, v_node, data in self.energy_graph.edges(data=True):
            child = v_node if parent.get(v_node) == u_node else u_node
            flow_value = float(subtree_sum[child])
            limit_value = float(data["limit"])
            branch_flow[self.edge_key((u_node, v_node))] = flow_value
            branch_excess.append(max(flow_value - limit_value, 0.0) / limit_value)

        voltage_proxy: Dict[int, float] = {}
        for node in self.energy_nodes:
            path = nx.shortest_path(self.energy_graph, source=0, target=node)
            voltage_drop = 0.0
            for node_a, node_b in zip(path[:-1], path[1:]):
                data = self.energy_graph[node_a][node_b]
                child = node_b if parent.get(node_b) == node_a else node_a
                flow_value = float(subtree_sum[child])
                voltage_drop += float(data["r"]) * flow_value / 350.0
            voltage_proxy[node] = 1.0 - voltage_drop

        overload_ratio = float(np.mean([1 if excess > 0 else 0 for excess in branch_excess]))
        mean_overload = float(np.mean(branch_excess))
        voltage_violation = float(np.mean([1 if (value < 0.93 or value > 1.05) else 0 for value in voltage_proxy.values()]))

        return {
            "node_load": node_load,
            "branch_flow": branch_flow,
            "overload_ratio": overload_ratio,
            "mean_overload": mean_overload,
            "voltage_violation": voltage_violation,
            "connected_load_ratio": 1.0,
        }

    def mobility_topology(
        self, edge_time: Dict[Tuple[int, int], float], residual_capacity: Dict[Tuple[int, int], float]
    ) -> Dict[str, np.ndarray]:
        """
        Build a graph-filtration summary over mobility stress.

        The filtration sweeps increasing admissibility thresholds and records:
        - Betti_0 (connected components),
        - Betti_1 (independent loops),
        - giant-component ratio.
        """
        thresholds = np.array([0.95, 1.05, 1.15, 1.25, 1.35, 1.50, 1.70, 2.00], dtype=float)
        normalized_time = {
            key: edge_time[key] / self.mobility_graph[key[0]][key[1]]["t0"]
            for key in edge_time
        }

        beta0_values: List[float] = []
        beta1_values: List[float] = []
        giant_component_values: List[float] = []

        for threshold in thresholds:
            threshold_graph = nx.Graph()
            threshold_graph.add_nodes_from(self.mobility_graph.nodes())

            for u_node, v_node in self.mobility_graph.edges():
                key = self.edge_key((u_node, v_node))
                if normalized_time[key] <= threshold and residual_capacity[key] > 0.08:
                    threshold_graph.add_edge(u_node, v_node)

            components = list(nx.connected_components(threshold_graph))
            beta0 = float(nx.number_connected_components(threshold_graph))
            beta1 = float(
                threshold_graph.number_of_edges()
                - threshold_graph.number_of_nodes()
                + beta0
            )
            giant_component = max((len(component) for component in components), default=1) / threshold_graph.number_of_nodes()

            beta0_values.append(beta0)
            beta1_values.append(max(beta1, 0.0))
            giant_component_values.append(float(giant_component))

        return {
            "thresholds": thresholds,
            "beta0": np.array(beta0_values, dtype=float),
            "beta1": np.array(beta1_values, dtype=float),
            "gcc": np.array(giant_component_values, dtype=float),
        }

    def topological_scores(
        self,
        topology: Dict[str, np.ndarray],
        baseline_topology: Dict[str, np.ndarray],
        service_metrics: Dict[str, float],
        energy_metrics: Dict[str, float],
    ) -> Tuple[float, float, float, float]:
        """
        Compare a scenario topology against the baseline topology.

        The score intentionally balances:
        - curve similarity,
        - invariant retention,
        - boundary proximity.

        This reflects the chapter's distinction between useful stress and
        implausible collapse.
        """
        weights = np.array([0.24, 0.20, 0.16, 0.13, 0.10, 0.08, 0.05, 0.04], dtype=float)
        weights = weights / weights.sum()

        diff_beta0 = np.sum(weights * np.abs(topology["beta0"] - baseline_topology["beta0"])) / max(
            np.max(baseline_topology["beta0"]), 1.0
        )
        diff_beta1 = np.sum(weights * np.abs(topology["beta1"] - baseline_topology["beta1"])) / max(
            np.max(baseline_topology["beta1"]), 1.0
        )
        diff_gcc = np.sum(weights * np.abs(topology["gcc"] - baseline_topology["gcc"]))

        similarity = float(
            np.clip(1.0 - (0.45 * diff_beta0 + 0.25 * diff_beta1 + 0.30 * diff_gcc) * 1.8, 0.0, 1.0)
        )

        min_gcc = float(np.min(topology["gcc"][1:]))
        invariants = float(
            0.40 * min_gcc
            + 0.30 * float(service_metrics["reachability_ratio"])
            + 0.30 * float(energy_metrics["connected_load_ratio"])
        )

        topology_validity = float(np.clip(0.50 * similarity + 0.50 * invariants, 0.0, 1.0))
        boundary_score = float(np.exp(-abs(topology_validity - 0.84) / 0.05))

        return similarity, invariants, topology_validity, boundary_score

    def simulate_scenario(
        self,
        program: Optional[ScenarioProgram],
        controller_name: str,
        learned_assets: Optional[Dict[str, object]] = None,
    ) -> Tuple[List[Dict[str, object]], int]:
        """
        Run a full scenario rollout over the time horizon.

        Returns
        -------
        timeseries:
            List of per-time-step dictionaries.
        peak_index:
            Index of the most stressful time step, used for topology scoring.
        """
        timeseries: List[Dict[str, object]] = []

        for time_index in range(self.horizon):
            modifiers = self.scenario_state(program, time_index)

            # Demand matrix for the current time step.
            od_matrix = self.gravity * self.demand_profile[time_index] * float(modifiers["demand_mult"])
            zone_bias = modifiers["zone_bias"]
            od_matrix = (od_matrix.T * zone_bias).T
            od_matrix = od_matrix * zone_bias
            od_matrix = od_matrix / od_matrix.sum() * self.gravity.sum() * self.demand_profile[time_index] * float(
                modifiers["demand_mult"]
            )

            # Mobility capacities after scenario degradation and control restoration.
            edge_t0: Dict[Tuple[int, int], float] = {}
            edge_cap: Dict[Tuple[int, int], float] = {}
            restoration = self.edge_capacity_restoration(
                controller_name=controller_name,
                scenario=program,
                time_index=time_index,
                learned_assets=learned_assets,
            )

            for u_node, v_node, data in self.mobility_graph.edges(data=True):
                key = self.edge_key((u_node, v_node))
                capacity_factor = float(modifiers["edge_cap_factor"][key]) * float(restoration[key])
                edge_cap[key] = max(float(data["capacity"]) * capacity_factor, 35.0)

                inflation = 1.0 + 0.25 * max(0.0, 1.0 - float(modifiers["edge_cap_factor"][key]))
                edge_t0[key] = float(data["t0"]) * inflation

            edge_flow, _ = self._route_assignment(edge_t0=edge_t0, edge_cap=edge_cap, od_matrix=od_matrix)
            edge_time, residual_capacity = self._compute_bpr_times(edge_flow=edge_flow, edge_cap=edge_cap)
            travel_matrix = self._all_pairs_travel(edge_time=edge_time)

            demand_total = float(od_matrix.sum())
            mobility_mean_tt = float(np.sum(travel_matrix * od_matrix) / (od_matrix.sum() + 1e-6))

            service_metrics = self.service_step(
                travel_matrix=travel_matrix,
                modifiers=modifiers,
                controller_name=controller_name,
                scenario=program,
                time_index=time_index,
                learned_assets=learned_assets,
            )

            energy_metrics = self.energy_step(
                modifiers=modifiers,
                controller_name=controller_name,
                scenario=program,
                time_index=time_index,
                demand_total=demand_total,
                learned_assets=learned_assets,
            )

            topology = self.mobility_topology(
                edge_time=edge_time,
                residual_capacity=residual_capacity,
            )

            timeseries.append(
                {
                    "t": time_index,
                    "demand_total": demand_total,
                    "mobility_mean_tt": mobility_mean_tt,
                    "mean_wait": float(service_metrics["mean_wait"]),
                    "access_dispersion": float(service_metrics["access_dispersion"]),
                    "reachability_ratio": float(service_metrics["reachability_ratio"]),
                    "overload_ratio": float(energy_metrics["overload_ratio"]),
                    "mean_overload": float(energy_metrics["mean_overload"]),
                    "voltage_violation": float(energy_metrics["voltage_violation"]),
                    "topology": topology,
                }
            )

        peak_index = int(
            np.argmax(
                [
                    row["mobility_mean_tt"] + 8.0 * row["overload_ratio"] + row["mean_wait"]
                    for row in timeseries
                ]
            )
        )
        return timeseries, peak_index

    def summarize_scenario(
        self,
        program: ScenarioProgram,
        controller_name: str = "Incumbent",
        learned_assets: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """
        Aggregate a rollout into the metrics used by the chapter.

        The summary is what eventually feeds:
        - the scenario bank,
        - the tables,
        - the plots,
        - the controller benchmarks.
        """
        timeseries, peak_index = self.simulate_scenario(
            program=program,
            controller_name=controller_name,
            learned_assets=learned_assets,
        )
        peak_row = timeseries[peak_index]

        topology_similarity, topology_invariants, topology_validity, boundary_score = self.topological_scores(
            topology=peak_row["topology"],
            baseline_topology=self.baseline_summary["topology"],
            service_metrics={"reachability_ratio": peak_row["reachability_ratio"]},
            energy_metrics={"connected_load_ratio": 1.0},
        )

        travel_ratio = float(np.mean([row["mobility_mean_tt"] for row in timeseries]) / self.baseline_summary["mobility_mean_tt"])
        wait_ratio = float(np.mean([row["mean_wait"] for row in timeseries]) / self.baseline_summary["mean_wait"])
        overload_mean = float(np.mean([row["mean_overload"] for row in timeseries]))
        access_dispersion = float(np.mean([row["access_dispersion"] for row in timeseries]))
        access_dispersion_ratio = float(access_dispersion / self.baseline_summary["access_dispersion"])

        # The energy score is normalized by a stress scale chosen for this toy city.
        energy_score = float(overload_mean / 0.04)

        # Global policy loss used in benchmarking and difficulty scoring.
        loss = float(
            0.45 * (travel_ratio - 1.0)
            + 0.20 * energy_score
            + 0.25 * (wait_ratio - 1.0)
            + 0.10 * (access_dispersion_ratio - 1.0)
        )

        feasible = int(
            peak_row["reachability_ratio"] >= 0.65
            and np.max([row["mean_overload"] for row in timeseries]) < 0.90
        )

        return {
            "family": program.family,
            "intensity": float(program.intensity),
            "travel_ratio": travel_ratio,
            "wait_ratio": wait_ratio,
            "energy_score": energy_score,
            "access_disp_ratio": access_dispersion_ratio,
            "loss": loss,
            "difficulty": loss,
            "topology_similarity": topology_similarity,
            "topology_invariants": topology_invariants,
            "topology_validity": topology_validity,
            "boundary_score": boundary_score,
            "reachability": float(peak_row["reachability_ratio"]),
            "overload_peak": float(np.max([row["mean_overload"] for row in timeseries])),
            "feasible": feasible,
            "peak_time": int(peak_row["t"]),
            "timeseries": timeseries,
        }

    # ------------------------------------------------------------------
    # Scenario-bank logic
    # ------------------------------------------------------------------
    def scenario_fingerprint(self, program: ScenarioProgram, summary: Dict[str, object]) -> np.ndarray:
        """
        Encode a scenario in a low-dimensional fingerprint space.

        This is the same abstraction used for:
        - novelty scoring,
        - bank coverage checks,
        - drift-aware retrieval.
        """
        facility_code: float
        if program.target_facility is None:
            facility_code = 0.0
        else:
            denominator = max(len(self.facilities) - 1, 1)
            facility_code = self.facilities.index(program.target_facility) / denominator

        return np.array(
            [
                self.family_to_idx[program.family] / max(len(self.families) - 1, 1),
                float(program.intensity),
                float(program.start / (self.horizon - 1)),
                float(program.duration / 4.0),
                float(len(program.target_edges) / 5.0),
                float(len(program.target_energy_nodes) / 4.0),
                facility_code,
                float(summary["travel_ratio"] - 1.0),
                float(summary["wait_ratio"] - 1.0),
                float(summary["energy_score"]),
                float(1.0 - summary["topology_validity"]),
            ],
            dtype=float,
        )

    @staticmethod
    def novelty_score(fingerprint: np.ndarray, bank_fingerprints: Sequence[np.ndarray]) -> float:
        """
        Score novelty by distance to the closest existing bank entry.
        """
        if not bank_fingerprints:
            return 1.0
        min_distance = min(float(np.linalg.norm(fingerprint - existing)) for existing in bank_fingerprints)
        return float(np.tanh(min_distance / 0.35))

    @staticmethod
    def composite_score(summary: Dict[str, object], novelty: float) -> float:
        """
        Score accepted scenarios with a multi-criterion objective.
        """
        return float(
            0.45 * float(summary["difficulty"])
            + 0.25 * float(summary["boundary_score"])
            + 0.20 * float(novelty)
            + 0.10 * float(summary["topology_validity"])
        )

    def generate_scenario_bank(
        self,
        num_generations: int = 6,
        population_size: int = 28,
        bank_size: int = 24,
        seed: int = 2026,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        """
        Generate an accepted scenario bank.

        Returns
        -------
        bank:
            Accepted scenarios with summaries and fingerprints.
        history:
            Full candidate history for diagnostics.
        """
        local_rng = np.random.default_rng(seed)
        population = [
            self.sample_scenario(seed=int(local_rng.integers(1, 1_000_000)))
            for _ in range(population_size)
        ]

        bank: List[Dict[str, object]] = []
        bank_fingerprints: List[np.ndarray] = []
        history: List[Dict[str, object]] = []

        for generation in range(num_generations):
            evaluations: List[Dict[str, object]] = []

            for candidate in population:
                summary = self.summarize_scenario(program=candidate, controller_name="Incumbent", learned_assets=None)
                fingerprint = self.scenario_fingerprint(program=candidate, summary=summary)
                novelty = self.novelty_score(fingerprint=fingerprint, bank_fingerprints=bank_fingerprints)

                accepted = int(
                    summary["feasible"] == 1
                    and summary["topology_validity"] >= 0.78
                    and novelty >= 0.12
                    and summary["difficulty"] >= 0.03
                )

                score = self.composite_score(summary=summary, novelty=novelty) if accepted else -1.0

                evaluations.append(
                    {
                        "program": candidate,
                        "summary": summary,
                        "fingerprint": fingerprint,
                        "novelty": novelty,
                        "accepted": accepted,
                        "score": score,
                        "generation": generation,
                    }
                )

            ranked = sorted(evaluations, key=lambda item: item["score"], reverse=True)

            # Insert accepted, non-duplicate scenarios into the bank.
            for item in ranked:
                if len(bank) >= bank_size:
                    break
                if item["accepted"] != 1:
                    continue

                fingerprint = item["fingerprint"]
                if bank_fingerprints:
                    nearest = min(np.linalg.norm(fingerprint - existing) for existing in bank_fingerprints)
                    if nearest < 0.14:
                        continue

                bank.append(item)
                bank_fingerprints.append(fingerprint)

            history.extend(evaluations)

            if len(bank) >= bank_size:
                break

            elites = [item for item in ranked if item["accepted"] == 1][:8]
            if len(elites) < 8:
                elites += ranked[: 8 - len(elites)]

            next_population: List[ScenarioProgram] = []
            for elite in elites:
                next_population.append(elite["program"])
                for _ in range(2):
                    next_population.append(
                        self.mutate_scenario(
                            program=elite["program"],
                            seed=int(local_rng.integers(1, 1_000_000)),
                        )
                    )

            while len(next_population) < population_size:
                next_population.append(
                    self.sample_scenario(seed=int(local_rng.integers(1, 1_000_000)))
                )

            population = next_population[:population_size]

        return bank, history

    def learn_assets_from_bank(self, bank: Sequence[Dict[str, object]]) -> Dict[str, object]:
        """
        Learn recurrently vulnerable assets from the accepted scenario bank.

        The result is used by the topology-aware controller.
        """
        edge_scores: Counter = Counter()
        energy_scores: Counter = Counter()
        facility_scores: Counter = Counter()
        family_scores: Counter = Counter()

        for item in bank:
            program: ScenarioProgram = item["program"]
            summary: Dict[str, object] = item["summary"]

            weight = float(summary["difficulty"]) * (1.0 + float(summary["boundary_score"]))
            family_scores[program.family] += weight

            for edge in program.target_edges:
                edge_scores[self.edge_key(edge)] += weight

            for node in program.target_energy_nodes:
                energy_scores[node] += weight

            if program.target_facility is not None:
                facility_scores[program.target_facility] += weight

        return {
            "critical_edges": [edge for edge, _ in edge_scores.most_common(4)],
            "critical_energy_nodes": [node for node, _ in energy_scores.most_common(4)],
            "critical_facilities": [facility for facility, _ in facility_scores.most_common(3)],
            "family_weights": dict(family_scores),
        }

    def evaluate_controllers(
        self,
        bank: Sequence[Dict[str, object]],
        learned_assets: Dict[str, object],
    ) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, object]]]]:
        """
        Evaluate the three benchmark controllers on the accepted scenario bank.
        """
        rows: List[Dict[str, object]] = []
        detail: Dict[str, List[Dict[str, object]]] = {}

        for controller_name in ["Incumbent", "Reactive", "TopologyAware"]:
            detail[controller_name] = []

            for item in bank:
                program: ScenarioProgram = item["program"]

                summary = self.summarize_scenario(
                    program=program,
                    controller_name=controller_name,
                    learned_assets=learned_assets if controller_name == "TopologyAware" else None,
                )

                detail[controller_name].append(summary)
                rows.append(
                    {
                        "controller": controller_name,
                        "family": program.family,
                        "loss": float(summary["loss"]),
                        "travel_ratio": float(summary["travel_ratio"]),
                        "wait_ratio": float(summary["wait_ratio"]),
                        "energy_score": float(summary["energy_score"]),
                        "access_disp_ratio": float(summary["access_disp_ratio"]),
                        "topology_validity": float(summary["topology_validity"]),
                        "scenario_id": int(program.seed),
                    }
                )

        return pd.DataFrame(rows), detail

    # ------------------------------------------------------------------
    # Drift monitoring
    # ------------------------------------------------------------------
    def apply_drift(self, epoch_name: str) -> None:
        """
        Apply one operating-epoch drift to the city and refresh the baseline.

        Three drifts are modeled:
        - Construction season,
        - EV adoption surge,
        - Service decentralization.
        """
        if epoch_name == "Construction season":
            for edge in [(3, 7), (6, 7), (5, 10)]:
                if self.mobility_graph.has_edge(*edge):
                    self.mobility_graph[edge[0]][edge[1]]["capacity"] *= 0.82
                    self.mobility_graph[edge[0]][edge[1]]["t0"] *= 1.10

            self.demand_profile = self.demand_profile * np.array(
                [1.00, 1.02, 1.05, 1.08, 1.06, 1.03, 1.01, 1.00], dtype=float
            )
            self.gravity = self.gravity * 1.03

        elif epoch_name == "EV adoption surge":
            for node in [2, 4, 6]:
                self.base_load[node] *= 1.22

            self.load_profile = self.load_profile * np.array(
                [1.00, 1.02, 1.05, 1.10, 1.18, 1.22, 1.15, 1.05], dtype=float
            )
            self.gravity = self.gravity * 1.04

        elif epoch_name == "Service decentralization":
            self.facility_service_rates[1] *= 0.85
            self.facility_service_rates[10] *= 1.22
            self.service_profile = self.service_profile * np.array(
                [0.95, 0.98, 1.00, 1.05, 1.08, 1.06, 1.00, 0.96], dtype=float
            )

            self.populations[[8, 9, 10, 11]] *= 1.08
            self.jobs[[0, 1, 2, 3]] *= 0.97

            gravity = np.outer(self.populations, self.jobs) / (self.distance_matrix ** 1.35)
            np.fill_diagonal(gravity, 0.0)
            gravity = gravity / gravity.sum() * 3350.0
            self.gravity = gravity

        else:
            raise ValueError(f"Unknown drift epoch: {epoch_name}")

        self._refresh_baseline()

    def mutate_bank(
        self,
        bank: Sequence[Dict[str, object]],
        seed: int,
        target_size: int = 12,
    ) -> List[Dict[str, object]]:
        """
        Create a hold-out bank by mutating accepted drift scenarios.

        This models the realistic case in which future scenarios are near but not
        identical to recently discovered regimes.
        """
        local_rng = np.random.default_rng(seed)
        holdout_bank: List[Dict[str, object]] = []
        if len(bank) == 0:
            return holdout_bank
        attempts = 0

        while len(holdout_bank) < target_size and attempts < target_size * 6:
            parent = bank[int(local_rng.integers(0, len(bank)))]
            candidate = self.mutate_scenario(
                program=parent["program"],
                seed=int(local_rng.integers(1, 1_000_000)),
            )

            if abs(candidate.intensity - parent["program"].intensity) < 0.02 and candidate.start == parent["program"].start:
                candidate.intensity = min(0.99, candidate.intensity + 0.08)

            summary = self.summarize_scenario(program=candidate, controller_name="Incumbent", learned_assets=None)
            fingerprint = self.scenario_fingerprint(program=candidate, summary=summary)

            novelty = self.novelty_score(
                fingerprint=fingerprint,
                bank_fingerprints=[self.scenario_fingerprint(item["program"], item["summary"]) for item in holdout_bank]
                if holdout_bank
                else [],
            )

            if summary["feasible"] == 1 and summary["topology_validity"] >= 0.76 and novelty > 0.05:
                holdout_bank.append(
                    {
                        "program": candidate,
                        "summary": summary,
                    }
                )

            attempts += 1

        return holdout_bank

    def bank_fingerprints(self, bank: Sequence[Dict[str, object]]) -> List[np.ndarray]:
        """
        Convenience wrapper to extract fingerprints for all bank entries.
        """
        return [self.scenario_fingerprint(item["program"], item["summary"]) for item in bank]

    @staticmethod
    def coverage_metrics(
        target_bank: Sequence[Dict[str, object]],
        reference_fingerprints: Sequence[np.ndarray],
        threshold: float = 0.36,
    ) -> Dict[str, float]:
        """
        Compute bank coverage and nearest-distance diagnostics.
        """
        target_fingerprints: List[np.ndarray] = []
        for item in target_bank:
            if "fingerprint" in item:
                target_fingerprints.append(item["fingerprint"])
            else:
                raise ValueError("Target bank entries must carry fingerprints when using this helper.")

        nearest_distances: List[float] = []
        covered = 0

        for fingerprint in target_fingerprints:
            nearest = min(float(np.linalg.norm(fingerprint - reference)) for reference in reference_fingerprints)
            nearest_distances.append(nearest)
            covered += int(nearest < threshold)

        return {
            "coverage": float(covered / max(len(target_fingerprints), 1)),
            "mean_nearest_distance": float(np.mean(nearest_distances)) if nearest_distances else float("inf"),
        }

    def evaluate_drift_epoch(
        self,
        epoch_name: str,
        original_bank: Sequence[Dict[str, object]],
        original_fingerprints: Sequence[np.ndarray],
        master_baseline_topology: Dict[str, np.ndarray],
        seed: int,
    ) -> Dict[str, object]:
        """
        Evaluate one structural-drift epoch.

        The method:
        1. restores the original city,
        2. applies one drift,
        3. generates a refreshed bank under the drifted city,
        4. creates a hold-out mutated bank,
        5. measures coverage before and after refresh.
        """
        self.restore_original_state()
        self.apply_drift(epoch_name=epoch_name)

        similarity, _, _, _ = self.topological_scores(
            topology=self.baseline_summary["topology"],
            baseline_topology=master_baseline_topology,
            service_metrics={"reachability_ratio": 1.0},
            energy_metrics={"connected_load_ratio": 1.0},
        )
        drift_score = float(1.0 - similarity)

        refresh_bank, _ = self.generate_scenario_bank(
            num_generations=4,
            population_size=24,
            bank_size=14,
            seed=seed,
        )

        holdout_bank = self.mutate_bank(
            bank=refresh_bank,
            seed=seed + 999,
            target_size=12,
        )

        refresh_bank_fingerprints = self.bank_fingerprints(refresh_bank)
        holdout_with_fingerprints = []
        for item in holdout_bank:
            holdout_with_fingerprints.append(
                {
                    "program": item["program"],
                    "summary": item["summary"],
                    "fingerprint": self.scenario_fingerprint(item["program"], item["summary"]),
                }
            )

        coverage_before = self.coverage_metrics(
            target_bank=holdout_with_fingerprints,
            reference_fingerprints=original_fingerprints,
            threshold=0.36,
        )
        coverage_after = self.coverage_metrics(
            target_bank=holdout_with_fingerprints,
            reference_fingerprints=list(original_fingerprints) + refresh_bank_fingerprints,
            threshold=0.36,
        )

        family_counter = Counter(item["program"].family for item in refresh_bank)
        dominant_family = family_counter.most_common(1)[0][0] if family_counter else ""

        return {
            "epoch": epoch_name,
            "drift_score": drift_score,
            "dominant_family": dominant_family,
            "coverage_before": float(coverage_before["coverage"]),
            "coverage_after": float(coverage_after["coverage"]),
            "distance_before": float(coverage_before["mean_nearest_distance"]),
            "distance_after": float(coverage_after["mean_nearest_distance"]),
            "coverage_lift": float(coverage_after["coverage"] - coverage_before["coverage"]),
            "distance_reduction": float(
                1.0 - coverage_after["mean_nearest_distance"] / coverage_before["mean_nearest_distance"]
            ),
            "refresh_bank_size": len(refresh_bank),
            "holdout_size": len(holdout_with_fingerprints),
        }

    # ------------------------------------------------------------------
    # Tables and plots
    # ------------------------------------------------------------------
    def build_table_1(self, bank: Sequence[Dict[str, object]]) -> pd.DataFrame:
        """
        Table 1 for Section 5.1:
        structural fragility and accepted scenario families.
        """
        records = []
        for index, item in enumerate(bank, start=1):
            program = item["program"]
            summary = item["summary"]
            records.append(
                {
                    "scenario_id": index,
                    "family": program.family,
                    "intensity": float(program.intensity),
                    "difficulty": float(summary["difficulty"]),
                    "topology_validity": float(summary["topology_validity"]),
                    "boundary_score": float(summary["boundary_score"]),
                    "novelty": float(item["novelty"]),
                    "travel_ratio": float(summary["travel_ratio"]),
                    "wait_ratio": float(summary["wait_ratio"]),
                    "energy_score": float(summary["energy_score"]),
                    "overload_peak": float(summary["overload_peak"]),
                }
            )

        bank_df = pd.DataFrame(records)
        table_1 = (
            bank_df.groupby("family")
            .agg(
                Accepted_Scenarios=("scenario_id", "count"),
                Mean_Difficulty=("difficulty", "mean"),
                Mean_Topology_Validity=("topology_validity", "mean"),
                Mean_Boundary_Score=("boundary_score", "mean"),
                Mean_Travel_Ratio=("travel_ratio", "mean"),
                Mean_Wait_Ratio=("wait_ratio", "mean"),
                Mean_Energy_Score=("energy_score", "mean"),
                Mean_Peak_Overload=("overload_peak", "mean"),
            )
            .reset_index()
        )
        return table_1, bank_df

    def build_table_2(self, controller_df: pd.DataFrame) -> pd.DataFrame:
        """
        Table 2 for Section 5.2:
        controller benchmark summary.

        The table is built defensively because some experimental runs may yield
        no accepted scenarios for one family, and degenerate runs may even
        produce sparse controller coverage. The export must remain valid in all
        those cases.
        """
        pivot = controller_df.pivot_table(index="scenario_id", columns="controller", values="loss")
        pivot = pivot.reindex(columns=CONTROLLER_ORDER)

        table_2 = (
            controller_df.groupby("controller")
            .agg(
                Mean_Loss=("loss", "mean"),
                P95_Loss=("loss", lambda series: np.quantile(series, 0.95)),
                Mean_Travel_Ratio=("travel_ratio", "mean"),
                Mean_Wait_Ratio=("wait_ratio", "mean"),
                Mean_Energy_Score=("energy_score", "mean"),
            )
            .reindex(CONTROLLER_ORDER)
            .reset_index()
            .rename(columns={"index": "controller"})
        )

        incumbent_series = table_2.loc[table_2["controller"] == "Incumbent", "Mean_Loss"]
        incumbent_mean_loss = float(incumbent_series.iloc[0]) if not incumbent_series.empty else float("nan")

        win_rates = {
            "Incumbent": 0.0,
            "Reactive": safe_controller_win_rate(pivot=pivot, controller_name="Reactive"),
            "TopologyAware": safe_controller_win_rate(pivot=pivot, controller_name="TopologyAware"),
        }

        table_2["Win_Rate_vs_Incumbent"] = table_2["controller"].map(win_rates)

        if not np.isfinite(incumbent_mean_loss) or abs(incumbent_mean_loss) <= 1e-12:
            table_2["Relative_Gain_vs_Incumbent"] = np.nan
        else:
            table_2["Relative_Gain_vs_Incumbent"] = 1.0 - table_2["Mean_Loss"] / incumbent_mean_loss

        return table_2

    @staticmethod
    def build_table_3(drift_df: pd.DataFrame) -> pd.DataFrame:
        """
        Table 3 for Section 5.3:
        drift monitoring and scenario-bank refresh diagnostics.
        """
        columns = [
            "epoch",
            "drift_score",
            "dominant_family",
            "coverage_before",
            "coverage_after",
            "distance_before",
            "distance_after",
            "coverage_lift",
            "distance_reduction",
        ]
        return drift_df[columns].copy()

    def plot_route_diagram(self, output_path: str) -> None:
        """
        Create the chapter roadmap figure requested for the introduction.
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 70)
        ax.axis("off")

        bands = [
            (0, 0, 22, 70, "#f1f7ff", "Urban data and problem framing"),
            (22, 0, 24, 70, "#f6fbf6", "Coupled digital twin"),
            (46, 0, 18, 70, "#fff8ef", "Absolute-Zero generation"),
            (64, 0, 18, 70, "#fcf4ff", "Topology and curation"),
            (82, 0, 18, 70, "#f7f7f7", "Benchmarking and operational learning"),
        ]
        for x_pos, y_pos, width, height, color, label in bands:
            ax.add_patch(Rectangle((x_pos, y_pos), width, height, facecolor=color, edgecolor="none", zorder=0))
            ax.text(x_pos + width / 2, 67.3, label, ha="center", va="center", fontsize=10.2, color="#4f4f4f", fontweight="bold")

        ax.text(
            50,
            64.3,
            "Research Roadmap for the Autonomous Urban Scenario Simulator",
            ha="center",
            va="center",
            fontsize=21.5,
            fontweight="bold",
            color="#17365d",
        )
        ax.text(
            50,
            61.0,
            "From heterogeneous urban observations to topology-verified scenario banks, controller benchmarking, and drift-aware shadow operation",
            ha="center",
            va="center",
            fontsize=11.3,
            color="#4d4d4d",
        )
        ax.text(
            50,
            57.8,
            "End-to-end workflow that connects the chapter's theory, methodology, architecture, empirical validation, and future operational use",
            ha="center",
            va="center",
            fontsize=9.8,
            color="#5b5b5b",
        )

        boxes = [
            dict(x=4, y=40, w=16, h=13, color="#5b9bd5", title="1. Data fusion", body="GIS, traffic sensors,\nutility records, service logs,\nweather, and policy calendars"),
            dict(x=24, y=40, w=18, h=13, color="#70ad47", title="2. Calibration", body="Temporal alignment,\nparameter estimation,\nand baseline validation"),
            dict(x=46, y=40, w=16, h=13, color="#ed7d31", title="3. Coupled urban twin", body="Mobility, energy, and public\nservices with cross-domain\nfeedback and accessibility cost"),
            dict(x=66, y=40, w=14, h=13, color="#a64d79", title="4. Scenario programs", body="Demand surges, incidents,\nload spikes, staffing shocks,\nand coupled perturbation rules"),
            dict(x=84, y=40, w=12, h=13, color="#7f7f7f", title="5. Rollout engine", body="Parallel simulation of\ncandidate futures"),
            dict(x=10, y=20, w=18, h=13, color="#5b9bd5", title="6. Topology checks", body="Betti curves, giant-component\nretention, reachability,\nand structural admissibility"),
            dict(x=33, y=20, w=18, h=13, color="#70ad47", title="7. Scenario bank", body="Accepted scenarios stored with\nfingerprints, metrics,\nseeds, and provenance"),
            dict(x=56, y=20, w=18, h=13, color="#ed7d31", title="8. Controller benchmark", body="Incumbent, reactive, and\ntopology-aware policies under\nbanked stress regimes"),
            dict(x=79, y=20, w=16, h=13, color="#a64d79", title="9. Shadow operation", body="Live-like replay,\nstructural drift monitoring,\nand targeted bank refresh"),
        ]

        for box in boxes:
            shadow = FancyBboxPatch(
                (box["x"] + 0.7, box["y"] - 0.7),
                box["w"],
                box["h"],
                boxstyle="round,pad=0.5,rounding_size=2.5",
                linewidth=0,
                facecolor="0.65",
                alpha=0.15,
                zorder=1,
            )
            ax.add_patch(shadow)

            patch = FancyBboxPatch(
                (box["x"], box["y"]),
                box["w"],
                box["h"],
                boxstyle="round,pad=0.5,rounding_size=2.5",
                linewidth=1.4,
                edgecolor="#ffffff",
                facecolor=box["color"],
                zorder=2,
            )
            ax.add_patch(patch)

            ax.text(
                box["x"] + box["w"] / 2,
                box["y"] + box["h"] - 2.5,
                box["title"],
                ha="center",
                va="center",
                fontsize=11.2,
                color="white",
                fontweight="bold",
                zorder=3,
            )
            ax.text(
                box["x"] + box["w"] / 2,
                box["y"] + box["h"] / 2 - 1.0,
                box["body"],
                ha="center",
                va="center",
                fontsize=9.1,
                color="white",
                linespacing=1.28,
                zorder=3,
            )

        def add_arrow(x1: float, y1: float, x2: float, y2: float, color: str, rad: float = 0.0, linewidth: float = 2.4) -> None:
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=linewidth,
                color=color,
                zorder=4,
            )
            ax.add_patch(arrow)

        add_arrow(20, 46.5, 24, 46.5, "#4d7fb3")
        add_arrow(42, 46.5, 46, 46.5, "#5c9944")
        add_arrow(62, 46.5, 66, 46.5, "#c26f28")
        add_arrow(80, 46.5, 84, 46.5, "#8f8f8f")
        add_arrow(90, 40, 76, 33, "#666666", rad=0.07)
        add_arrow(19, 33, 19, 39.5, "#4d7fb3")
        add_arrow(28, 26.5, 33, 26.5, "#5c9944")
        add_arrow(33, 26.5, 28, 26.5, "#5c9944")
        add_arrow(51, 26.5, 56, 26.5, "#c26f28")
        add_arrow(56, 26.5, 51, 26.5, "#c26f28")
        add_arrow(74, 26.5, 79, 26.5, "#9a5576")
        add_arrow(95, 26.5, 95, 46.5, "#8f8f8f")

        # Continuous-learning loop
        loop = FancyArrowPatch(
            (87, 20),
            (12, 40),
            connectionstyle="arc3,rad=0.42",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.6,
            linestyle="--",
            color="#cc4e4e",
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(loop)
        ax.text(
            50,
            16.6,
            "Continuous-learning loop: drift detection launches recalibration and new scenario generation,\nthen returns the updated bank to benchmarking and shadow operation",
            ha="center",
            va="center",
            fontsize=10.4,
            color="#8a2d2d",
            fontweight="bold",
        )

        ribbons = [
            (7, 6.3, 21, 6.8, "#d9e8f6", "#2f5597", "Section 2  |  Theory"),
            (31, 6.3, 21, 6.8, "#e2f0d9", "#548235", "Section 3  |  Methodology"),
            (55, 6.3, 18, 6.8, "#fbe5d6", "#c55a11", "Section 4  |  Architecture"),
            (76, 6.3, 20, 6.8, "#eadcf1", "#7f3f98", "Sections 5-7  |  Validation and future work"),
        ]
        for x_pos, y_pos, width, height, facecolor, edgecolor, label in ribbons:
            patch = FancyBboxPatch(
                (x_pos, y_pos),
                width,
                height,
                boxstyle="round,pad=0.4,rounding_size=1.8",
                linewidth=1.1,
                edgecolor=edgecolor,
                facecolor=facecolor,
                zorder=2,
            )
            ax.add_patch(patch)
            ax.text(
                x_pos + width / 2,
                y_pos + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=10.0,
                color=edgecolor,
                fontweight="bold",
            )

        fig.tight_layout(pad=0.6)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_section_5_1(
        bank_df: pd.DataFrame,
        output_dir: str,
    ) -> None:
        """
        Create the Section 5.1 figures.
        """
        family_palette = {
            "ClinicOutage": "#1f77b4",
            "CorridorIncident": "#ff7f0e",
            "EVChargingShock": "#2ca02c",
            "HeatwaveLoad": "#d62728",
            "StadiumEvent": "#9467bd",
            "StormCascade": "#8c564b",
        }

        # Figure 2: difficulty vs topology validity.
        fig, axis = plt.subplots(figsize=(8.6, 5.6))
        for family in sorted(bank_df["family"].unique()):
            subset = bank_df[bank_df["family"] == family]
            axis.scatter(
                subset["topology_validity"],
                subset["difficulty"],
                s=120 + 220 * subset["boundary_score"],
                alpha=0.82,
                label=family,
                c=family_palette[family],
                edgecolor="white",
                linewidth=0.9,
            )

        axis.axvspan(0.78, 0.86, color="#f8d5a8", alpha=0.35, label="Boundary regime")
        axis.axvline(0.78, color="#bb7722", linestyle="--", linewidth=1.2)
        axis.set_xlabel("Topology validity score")
        axis.set_ylabel("Scenario difficulty")
        axis.set_title("Accepted scenarios: difficulty versus structural admissibility")
        axis.grid(True, alpha=0.18)
        axis.legend(ncols=2, fontsize=8, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "figure_2_difficulty_vs_topology.png"), dpi=260, bbox_inches="tight")
        plt.close(fig)

        # Figure 3: burden heatmap by family.
        heat_df = bank_df.groupby("family")[["travel_ratio", "wait_ratio", "energy_score", "topology_validity"]].mean().copy()
        heat_df["travel_burden"] = heat_df["travel_ratio"] - 1.0
        heat_df["service_burden"] = heat_df["wait_ratio"] - 1.0
        heat_df["topology_deviation"] = 1.0 - heat_df["topology_validity"]
        heat_values = heat_df[["travel_burden", "service_burden", "energy_score", "topology_deviation"]]
        heat_normalized = (heat_values - heat_values.min()) / (heat_values.max() - heat_values.min() + 1e-9)

        fig, axis = plt.subplots(figsize=(8.6, 4.8))
        image = axis.imshow(heat_normalized.values, aspect="auto", cmap="YlOrRd")
        axis.set_yticks(range(len(heat_normalized.index)))
        axis.set_yticklabels(heat_normalized.index)
        axis.set_xticks(range(len(heat_normalized.columns)))
        axis.set_xticklabels(["Travel\nburden", "Service\nburden", "Energy\nstress", "Topology\ndeviation"])

        for i_row in range(heat_normalized.shape[0]):
            for j_col in range(heat_normalized.shape[1]):
                axis.text(j_col, i_row, f"{heat_values.iloc[i_row, j_col]:.2f}", ha="center", va="center", fontsize=8)

        axis.set_title("Mean normalized burden profile by accepted scenario family")
        colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
        colorbar.set_label("Column-wise normalized intensity")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "figure_3_family_burden_heatmap.png"), dpi=260, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_section_5_2(
        controller_df: pd.DataFrame,
        output_dir: str,
    ) -> None:
        """
        Create the Section 5.2 figures.

        The heatmap is generated with reindexing rather than `.loc[...]` so
        that absent families remain visible as missing cells instead of raising
        `KeyError`.
        """
        family_controller = ordered_controller_family_loss_matrix(controller_df)

        # Figure 4: mean loss heatmap by family and controller.
        fig, axis = plt.subplots(figsize=(7.8, 4.8))
        image = nan_aware_imshow(axis, family_controller, cmap_name="Blues")
        axis.set_yticks(range(len(family_controller.index)))
        axis.set_yticklabels(family_controller.index)
        axis.set_xticks(range(len(family_controller.columns)))
        axis.set_xticklabels(["Incumbent", "Reactive", "Topology-aware"])
        annotate_dataframe_cells(axis, family_controller, fontsize=8)

        axis.set_title("Mean policy loss by scenario family and controller")
        colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
        colorbar.set_label("Loss")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "figure_4_controller_family_heatmap.png"), dpi=260, bbox_inches="tight")
        plt.close(fig)

        # Figure 5: loss distribution across the scenario bank.
        fig, axis = plt.subplots(figsize=(7.8, 5.0))
        data = [
            controller_df[controller_df["controller"] == controller_name]["loss"].values
            for controller_name in CONTROLLER_ORDER
        ]

        boxplot = axis.boxplot(
            data,
            patch_artist=True,
            tick_labels=["Incumbent", "Reactive", "Topology-aware"],
            medianprops=dict(color="black", linewidth=1.3),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )

        colors = ["#7ea6e0", "#8ac6a4", "#f3c178"]
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        for index, controller_name in enumerate(CONTROLLER_ORDER, start=1):
            values = controller_df[controller_df["controller"] == controller_name]["loss"].values
            x_values = np.random.default_rng(10 + index).normal(index, 0.04, size=len(values))
            axis.scatter(x_values, values, s=14, alpha=0.45, color="#444444")

        axis.set_ylabel("Scenario loss")
        axis.set_title("Distribution of policy losses across the accepted scenario bank")
        axis.grid(True, axis="y", alpha=0.18)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "figure_5_loss_distribution.png"), dpi=260, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_section_5_3(
        drift_df: pd.DataFrame,
        output_dir: str,
    ) -> None:
        """
        Create the Section 5.3 figures.
        """
        # Figure 6: structural drift monitor.
        full_epochs = ["Baseline"] + drift_df["epoch"].tolist()
        drift_scores = [0.0] + drift_df["drift_score"].tolist()

        fig, axis = plt.subplots(figsize=(8.0, 4.6))
        x_values = np.arange(len(full_epochs))
        axis.plot(x_values, drift_scores, marker="o", linewidth=2.2, color="#2f6db3")

        for x_value, y_value in zip(x_values, drift_scores):
            axis.text(x_value, y_value + 0.006, f"{y_value:.2f}", ha="center", va="bottom", fontsize=8)

        axis.set_xticks(x_values)
        axis.set_xticklabels(["Baseline", "Construction", "EV adoption", "Service\ndecentralization"])
        axis.set_ylabel("Structural drift score")
        axis.set_title("Structural drift monitor across operating epochs")
        axis.grid(True, axis="y", alpha=0.18)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "figure_6_structural_drift.png"), dpi=260, bbox_inches="tight")
        plt.close(fig)

        # Figure 7: bank coverage before/after refresh.
        fig, axis = plt.subplots(figsize=(8.2, 4.8))
        x_positions = np.arange(len(drift_df))
        width = 0.34

        axis.bar(x_positions - width / 2, drift_df["coverage_before"], width=width, label="Before refresh", color="#b8cbe5")
        axis.bar(x_positions + width / 2, drift_df["coverage_after"], width=width, label="After refresh", color="#4f81bd")

        for x_value, before_value, after_value in zip(x_positions, drift_df["coverage_before"], drift_df["coverage_after"]):
            axis.text(x_value - width / 2, before_value + 0.015, f"{before_value:.2f}", ha="center", va="bottom", fontsize=8)
            axis.text(x_value + width / 2, after_value + 0.015, f"{after_value:.2f}", ha="center", va="bottom", fontsize=8)

        axis.set_xticks(x_positions)
        axis.set_xticklabels(["Construction", "EV adoption", "Service\ndecentralization"])
        axis.set_ylabel("Hold-out coverage")
        axis.set_ylim(0.0, max(0.72, float(drift_df["coverage_after"].max()) + 0.10))
        axis.set_title("Scenario-bank coverage before and after refresh under drift")
        axis.legend(frameon=False)
        axis.grid(True, axis="y", alpha=0.18)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "figure_7_coverage_refresh.png"), dpi=260, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Master execution method
    # ------------------------------------------------------------------
    def run_all(self, output_dir: str) -> Dict[str, str]:
        """
        Generate the complete package of tables and figures.

        This is the single entry point expected by chapter users.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Ensure the original state is active before building the main bank.
        self.restore_original_state()

        # Main scenario-bank generation for Sections 5.1 and 5.2.
        scenario_bank, generation_history = self.generate_scenario_bank(
            num_generations=6,
            population_size=28,
            bank_size=24,
            seed=self.master_seed,
        )

        learned_assets = self.learn_assets_from_bank(scenario_bank)
        controller_df, _ = self.evaluate_controllers(
            bank=scenario_bank,
            learned_assets=learned_assets,
        )

        # Tables
        table_1, bank_df = self.build_table_1(scenario_bank)
        table_2 = self.build_table_2(controller_df)

        master_baseline_topology = copy.deepcopy(self.baseline_summary["topology"])
        original_fingerprints = self.bank_fingerprints(scenario_bank)

        drift_rows = []
        for index, epoch_name in enumerate(
            ["Construction season", "EV adoption surge", "Service decentralization"]
        ):
            drift_rows.append(
                self.evaluate_drift_epoch(
                    epoch_name=epoch_name,
                    original_bank=scenario_bank,
                    original_fingerprints=original_fingerprints,
                    master_baseline_topology=master_baseline_topology,
                    seed=3100 + index,
                )
            )

        drift_df = pd.DataFrame(drift_rows)
        table_3 = self.build_table_3(drift_df)

        # Save CSV tables.
        table_1_path = os.path.join(output_dir, "table_1_structural_fragility_summary.csv")
        table_2_path = os.path.join(output_dir, "table_2_controller_benchmark.csv")
        table_3_path = os.path.join(output_dir, "table_3_drift_refresh_summary.csv")
        bank_detail_path = os.path.join(output_dir, "accepted_scenario_bank_detail.csv")
        controller_detail_path = os.path.join(output_dir, "controller_benchmark_detail.csv")

        table_1.to_csv(table_1_path, index=False)
        table_2.to_csv(table_2_path, index=False)
        table_3.to_csv(table_3_path, index=False)
        bank_df.to_csv(bank_detail_path, index=False)
        controller_df.to_csv(controller_detail_path, index=False)

        # Save metadata and learned assets.
        metadata_path = os.path.join(output_dir, "run_metadata.json")
        metadata = {
            "master_seed": self.master_seed,
            "families": self.families,
            "learned_assets": learned_assets,
            "bank_size": len(scenario_bank),
            "history_size": len(generation_history),
            "drift_epochs": drift_df.to_dict(orient="records"),
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        # Save figures.
        route_diagram_path = os.path.join(output_dir, "figure_1_research_roadmap_autonomous_urban_simulator.png")
        self.plot_route_diagram(route_diagram_path)
        self.plot_section_5_1(bank_df=bank_df, output_dir=output_dir)
        self.plot_section_5_2(controller_df=controller_df, output_dir=output_dir)
        self.plot_section_5_3(drift_df=drift_df, output_dir=output_dir)

        # Restore the original city so repeated calls start from a clean state.
        self.restore_original_state()

        return {
            "table_1": table_1_path,
            "table_2": table_2_path,
            "table_3": table_3_path,
            "bank_detail": bank_detail_path,
            "controller_detail": controller_detail_path,
            "metadata": metadata_path,
            "figure_1": route_diagram_path,
            "figure_2": os.path.join(output_dir, "figure_2_difficulty_vs_topology.png"),
            "figure_3": os.path.join(output_dir, "figure_3_family_burden_heatmap.png"),
            "figure_4": os.path.join(output_dir, "figure_4_controller_family_heatmap.png"),
            "figure_5": os.path.join(output_dir, "figure_5_loss_distribution.png"),
            "figure_6": os.path.join(output_dir, "figure_6_structural_drift.png"),
            "figure_7": os.path.join(output_dir, "figure_7_coverage_refresh.png"),
        }



# =====================================================================
# Algebraic-topology extension of the base simulator
# =====================================================================

class TopologicalAutonomousUrbanScenarioSimulator(AutonomousUrbanScenarioSimulator):
    """
    Extension of the base simulator that replaces the lightweight topology
    surrogates with an explicit algebraic-topology workflow based on:

    - Ripser / Persim for persistent homology and diagram distances,
    - TopoNetX for simplicial-complex lifting and Hodge diagnostics,
    - KeplerMapper for scenario-bank shape visualization,
    - torch for nonlinear scenario-bank embeddings.

    The base simulator is preserved because it already encodes the coupled
    mobility-energy-service dynamics required by the article. This subclass
    augments the topological, representational, and export layers so that the
    generated outputs support Sections 5.1, 5.2, and 5.3 more directly.
    """

    def __init__(self, master_seed: int = 2026) -> None:
        check_dependencies()
        set_global_seed(master_seed)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.module_rng = np.random.default_rng(master_seed)
        super().__init__(master_seed=master_seed)

    # ------------------------------------------------------------------
    # Baseline refresh with explicit topological reports
    # ------------------------------------------------------------------
    def _refresh_baseline(self) -> None:
        """
        Recompute the baseline city summary and attach a full TDA report.

        The baseline report becomes the structural reference used by:
        - scenario admissibility,
        - drift measurements,
        - persistence-diagram comparisons,
        - higher-order TopoNetX metrics.
        """
        baseline_timeseries, peak_index = self.simulate_scenario(
            program=None,
            controller_name="Incumbent",
            learned_assets=None,
        )
        peak_row = baseline_timeseries[peak_index]
        baseline_topology = self.topology_report_from_peak_row(peak_row)

        self.baseline_summary = {
            "mobility_mean_tt": float(np.mean([row["mobility_mean_tt"] for row in baseline_timeseries])),
            "mean_wait": float(np.mean([row["mean_wait"] for row in baseline_timeseries])),
            "access_dispersion": float(np.mean([row["access_dispersion"] for row in baseline_timeseries])),
            "topology": baseline_topology,
            "peak_row": peak_row,
            "peak_time": int(peak_row["t"]),
        }

    # ------------------------------------------------------------------
    # Extended simulator: store richer state for TDA
    # ------------------------------------------------------------------
    def simulate_scenario(
        self,
        program: Optional[ScenarioProgram],
        controller_name: str,
        learned_assets: Optional[Dict[str, object]] = None,
    ) -> Tuple[List[Dict[str, object]], int]:
        """
        Run a full scenario rollout and keep the richer state required by the
        topological-analysis stage.

        Compared with the base implementation, each time slice now stores:
        - the full all-pairs travel matrix,
        - edge times and residual capacities,
        - access costs,
        - node loads and branch flows,
        - facility waits.

        These additional artifacts allow persistent homology and simplicial
        diagnostics to be computed on the peak stress snapshot.
        """
        timeseries: List[Dict[str, object]] = []

        for time_index in range(self.horizon):
            modifiers = self.scenario_state(program, time_index)

            od_matrix = self.gravity * self.demand_profile[time_index] * float(modifiers["demand_mult"])
            zone_bias = modifiers["zone_bias"]
            od_matrix = (od_matrix.T * zone_bias).T
            od_matrix = od_matrix * zone_bias
            od_matrix = od_matrix / od_matrix.sum() * self.gravity.sum() * self.demand_profile[time_index] * float(
                modifiers["demand_mult"]
            )

            edge_t0: Dict[Tuple[int, int], float] = {}
            edge_cap: Dict[Tuple[int, int], float] = {}
            restoration = self.edge_capacity_restoration(
                controller_name=controller_name,
                scenario=program,
                time_index=time_index,
                learned_assets=learned_assets,
            )

            for u_node, v_node, data in self.mobility_graph.edges(data=True):
                key = self.edge_key((u_node, v_node))
                capacity_factor = float(modifiers["edge_cap_factor"][key]) * float(restoration[key])
                edge_cap[key] = max(float(data["capacity"]) * capacity_factor, 35.0)

                inflation = 1.0 + 0.25 * max(0.0, 1.0 - float(modifiers["edge_cap_factor"][key]))
                edge_t0[key] = float(data["t0"]) * inflation

            edge_flow, route_usage = self._route_assignment(edge_t0=edge_t0, edge_cap=edge_cap, od_matrix=od_matrix)
            edge_time, residual_capacity = self._compute_bpr_times(edge_flow=edge_flow, edge_cap=edge_cap)
            travel_matrix = self._all_pairs_travel(edge_time=edge_time)

            demand_total = float(od_matrix.sum())
            mobility_mean_tt = float(np.sum(travel_matrix * od_matrix) / (od_matrix.sum() + 1e-6))

            service_metrics = self.service_step(
                travel_matrix=travel_matrix,
                modifiers=modifiers,
                controller_name=controller_name,
                scenario=program,
                time_index=time_index,
                learned_assets=learned_assets,
            )

            energy_metrics = self.energy_step(
                modifiers=modifiers,
                controller_name=controller_name,
                scenario=program,
                time_index=time_index,
                demand_total=demand_total,
                learned_assets=learned_assets,
            )

            curve_topology = self.mobility_topology(
                edge_time=edge_time,
                residual_capacity=residual_capacity,
            )

            timeseries.append(
                {
                    "t": time_index,
                    "demand_total": demand_total,
                    "od_matrix": od_matrix.copy(),
                    "mobility_mean_tt": mobility_mean_tt,
                    "travel_matrix": travel_matrix.copy(),
                    "edge_time": dict(edge_time),
                    "residual_capacity": dict(residual_capacity),
                    "edge_flow": dict(edge_flow),
                    "route_usage": route_usage,
                    "mean_wait": float(service_metrics["mean_wait"]),
                    "facility_waits": dict(service_metrics["facility_waits"]),
                    "facility_arrival": dict(service_metrics["facility_arrival"]),
                    "access_costs": service_metrics["access_costs"].copy(),
                    "access_dispersion": float(service_metrics["access_dispersion"]),
                    "reachability_ratio": float(service_metrics["reachability_ratio"]),
                    "overload_ratio": float(energy_metrics["overload_ratio"]),
                    "mean_overload": float(energy_metrics["mean_overload"]),
                    "voltage_violation": float(energy_metrics["voltage_violation"]),
                    "node_load": dict(energy_metrics["node_load"]),
                    "branch_flow": dict(energy_metrics["branch_flow"]),
                    "connected_load_ratio": float(energy_metrics["connected_load_ratio"]),
                    "topology_curve": curve_topology,
                }
            )

        peak_index = int(
            np.argmax(
                [
                    row["mobility_mean_tt"] + 8.0 * row["overload_ratio"] + row["mean_wait"]
                    for row in timeseries
                ]
            )
        )
        return timeseries, peak_index

    # ------------------------------------------------------------------
    # Cross-domain point-cloud builders
    # ------------------------------------------------------------------
    def zone_positions(self) -> np.ndarray:
        """
        Return the zone coordinates as a dense array.

        These coordinates are reused in multiple topological summaries.
        """
        return np.array([self.coords[i] for i in range(len(self.coords))], dtype=float)

    def energy_node_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Approximate a geometric position for each feeder node.

        Since the stylized feeder graph is not embedded explicitly, the method
        places each energy node at the centroid of the urban zones it serves.
        """
        positions: Dict[int, Tuple[float, float]] = {}
        zone_positions = self.zone_positions()
        for energy_node in self.energy_nodes:
            assigned_zones = [zone for zone, node in self.zone_to_energy_node.items() if node == energy_node]
            if assigned_zones:
                centroid = np.mean(zone_positions[assigned_zones], axis=0)
                positions[energy_node] = (float(centroid[0]), float(centroid[1]))
            else:
                positions[energy_node] = (float(energy_node), 0.0)
        return positions

    def zone_point_cloud(self, peak_row: Dict[str, object]) -> np.ndarray:
        """
        Build a cross-domain point cloud at zone level.

        Each point mixes:
        - geometric position,
        - mean travel burden,
        - access cost,
        - local energy load,
        - population share,
        - graph degree.

        This realizes the chapter's idea that topology should reflect the
        coupled state space of mobility, energy, and service access rather than
        one subsystem alone.
        """
        positions = self.zone_positions()
        degree_map = dict(self.mobility_graph.degree())
        max_degree = max(degree_map.values())
        max_load = max(float(value) for value in peak_row["node_load"].values())

        point_rows = []
        for zone in range(len(self.coords)):
            x_coord, y_coord = positions[zone]
            mean_travel = float(np.mean(peak_row["travel_matrix"][zone]))
            access_cost = float(peak_row["access_costs"][zone])
            energy_node = self.zone_to_energy_node[zone]
            local_load = float(peak_row["node_load"][energy_node])
            population_share = float(self.populations[zone] / self.populations.sum())
            degree_ratio = float(degree_map[zone] / max(max_degree, 1))

            point_rows.append(
                [
                    x_coord,
                    y_coord,
                    mean_travel,
                    access_cost,
                    local_load / max(max_load, 1.0),
                    population_share,
                    degree_ratio,
                ]
            )

        cloud = np.asarray(point_rows, dtype=float)
        return StandardScaler().fit_transform(cloud)

    def energy_point_cloud(self, peak_row: Dict[str, object]) -> np.ndarray:
        """
        Build an energy-domain point cloud from feeder-node states.

        The point cloud combines:
        - centroid position,
        - normalized load,
        - branch stress around the node,
        - node degree.

        This gives Ripser a compact but informative geometric representation of
        feeder stress under each scenario.
        """
        positions = self.energy_node_positions()
        max_load = max(float(value) for value in peak_row["node_load"].values())
        max_degree = max(dict(self.energy_graph.degree()).values())

        point_rows = []
        for node in self.energy_nodes:
            x_coord, y_coord = positions[node]
            load_value = float(peak_row["node_load"][node]) / max(max_load, 1.0)
            degree_ratio = float(self.energy_graph.degree(node) / max(max_degree, 1))

            incident_stress = []
            for neighbor in self.energy_graph.neighbors(node):
                edge = self.edge_key((node, neighbor))
                branch_flow = abs(float(peak_row["branch_flow"][edge]))
                limit = float(self.energy_graph[node][neighbor]["limit"])
                incident_stress.append(branch_flow / max(limit, 1e-9))
            mean_incident_stress = float(np.mean(incident_stress)) if incident_stress else 0.0

            point_rows.append(
                [
                    x_coord,
                    y_coord,
                    load_value,
                    degree_ratio,
                    mean_incident_stress,
                ]
            )

        cloud = np.asarray(point_rows, dtype=float)
        return StandardScaler().fit_transform(cloud)

    # ------------------------------------------------------------------
    # Persistent homology and higher-order topology
    # ------------------------------------------------------------------
    def persistence_summary(self, point_cloud: np.ndarray) -> Dict[str, object]:
        """
        Compute persistent homology for a point cloud.

        Ripser returns persistence diagrams for H0 and H1. The method also
        computes persistent entropies so the downstream analysis has both
        geometric and scalar summaries.
        """
        ph_result = ripser(point_cloud, maxdim=1)
        diagrams = ph_result["dgms"]
        if len(diagrams) == 1:
            diagrams = [diagrams[0], np.zeros((0, 2), dtype=float)]

        return {
            "diagrams": diagrams,
            "entropy_h0": persistent_entropy(diagrams[0]),
            "entropy_h1": persistent_entropy(diagrams[1]),
            "count_h1": int(finite_diagram(diagrams[1]).shape[0]),
        }

    def toponetx_hodge_report(self, peak_row: Dict[str, object]) -> Dict[str, float]:
        """
        Lift the thresholded mobility graph to a simplicial complex and compute
        Hodge-style diagnostics using TopoNetX incidence matrices.

        The method uses the following logic:
        1. Build a thresholded mobility graph that keeps serviceable edges.
        2. Lift the graph to a clique complex up to rank 2.
        3. Compute incidence matrices B1 and B2.
        4. Form the rank-1 Hodge Laplacian:
               L1 = B1^T B1 + B2 B2^T
        5. Summarize the resulting spectrum.
        """
        threshold_graph = nx.Graph()
        threshold_graph.add_nodes_from(self.mobility_graph.nodes())

        normalized_times: List[float] = []
        edge_records: List[Tuple[Tuple[int, int], float, float]] = []
        for u_node, v_node in self.mobility_graph.edges():
            key = self.edge_key((u_node, v_node))
            normalized_time = float(peak_row["edge_time"][key] / self.mobility_graph[u_node][v_node]["t0"])
            residual = float(peak_row["residual_capacity"][key])
            normalized_times.append(normalized_time)
            edge_records.append((key, normalized_time, residual))

        if normalized_times:
            time_cutoff = float(np.quantile(normalized_times, 0.70))
        else:
            time_cutoff = 1.0

        for key, normalized_time, residual in edge_records:
            if normalized_time <= time_cutoff and residual > 0.06:
                threshold_graph.add_edge(*key, weight=normalized_time)

        if threshold_graph.number_of_edges() == 0:
            threshold_graph = self.mobility_graph.copy()
            for u_node, v_node in threshold_graph.edges():
                key = self.edge_key((u_node, v_node))
                threshold_graph[u_node][v_node]["weight"] = float(
                    peak_row["edge_time"][key] / self.mobility_graph[u_node][v_node]["t0"]
                )

        simplicial_complex = graph_to_clique_complex(threshold_graph, max_rank=2)

        if getattr(simplicial_complex, "dim", 0) >= 1:
            b1 = sparse_to_dense(simplicial_complex.incidence_matrix(1))
            if b1.ndim == 1:
                b1 = b1.reshape(-1, 1)
        else:
            b1 = np.zeros((threshold_graph.number_of_nodes(), 0), dtype=float)

        if getattr(simplicial_complex, "dim", 0) >= 2:
            b2 = sparse_to_dense(simplicial_complex.incidence_matrix(2))
            if b2.ndim == 1:
                b2 = b2.reshape(-1, 1)
        else:
            b2 = np.zeros((b1.shape[1], 0), dtype=float)

        if b1.size == 0:
            l1 = np.zeros((0, 0), dtype=float)
        else:
            l1 = b1.T @ b1
            if b2.size > 0:
                l1 = l1 + b2 @ b2.T

        if l1.size == 0:
            eigenvalues = np.zeros(0, dtype=float)
            harmonic_count = 0
            spectral_gap = 0.0
        else:
            eigenvalues = np.linalg.eigvalsh(l1)
            harmonic_count = int(np.sum(np.isclose(eigenvalues, 0.0, atol=1e-8)))
            positive = eigenvalues[eigenvalues > 1e-8]
            spectral_gap = float(np.min(positive)) if positive.size else 0.0

        triangle_count = sum(1 for clique in nx.enumerate_all_cliques(threshold_graph) if len(clique) == 3)
        component_count = nx.number_connected_components(threshold_graph)
        gcc_ratio = max((len(component) for component in nx.connected_components(threshold_graph)), default=1) / max(
            threshold_graph.number_of_nodes(), 1
        )

        return {
            "triangle_count": int(triangle_count),
            "harmonic_count": int(harmonic_count),
            "spectral_gap": float(spectral_gap),
            "component_count": float(component_count),
            "gcc_ratio": float(gcc_ratio),
            "edge_count_threshold_graph": float(threshold_graph.number_of_edges()),
        }

    def topology_report_from_peak_row(self, peak_row: Dict[str, object]) -> Dict[str, object]:
        """
        Build the full topological report for a peak scenario snapshot.

        The report contains three complementary views:
        - the original graph-filtration summary from the base simulator,
        - persistent-homology summaries on cross-domain point clouds,
        - higher-order simplicial diagnostics via TopoNetX.
        """
        curve_report = peak_row["topology_curve"]
        zone_cloud = self.zone_point_cloud(peak_row)
        energy_cloud = self.energy_point_cloud(peak_row)
        zone_ph = self.persistence_summary(zone_cloud)
        energy_ph = self.persistence_summary(energy_cloud)
        hodge_report = self.toponetx_hodge_report(peak_row)

        return {
            "curve": curve_report,
            "zone_cloud": zone_cloud,
            "energy_cloud": energy_cloud,
            "zone_diagrams": zone_ph["diagrams"],
            "energy_diagrams": energy_ph["diagrams"],
            "zone_entropy_h0": float(zone_ph["entropy_h0"]),
            "zone_entropy_h1": float(zone_ph["entropy_h1"]),
            "energy_entropy_h0": float(energy_ph["entropy_h0"]),
            "zone_h1_count": int(zone_ph["count_h1"]),
            "triangle_count": int(hodge_report["triangle_count"]),
            "harmonic_count": int(hodge_report["harmonic_count"]),
            "spectral_gap": float(hodge_report["spectral_gap"]),
            "component_count": float(hodge_report["component_count"]),
            "gcc_ratio_hodge": float(hodge_report["gcc_ratio"]),
            "edge_count_threshold_graph": float(hodge_report["edge_count_threshold_graph"]),
        }

    def safe_persim_distance(self, left: np.ndarray, right: np.ndarray, metric: str = "wasserstein") -> float:
        """
        Compare two persistence diagrams robustly.

        Persim's distance functions are used directly, but the method guards
        against empty diagrams by replacing them with diagonal fallback points.
        """
        left_safe = diagram_with_fallback(left)
        right_safe = diagram_with_fallback(right)
        if metric == "bottleneck":
            return float(bottleneck(left_safe, right_safe))
        return float(wasserstein(left_safe, right_safe))

    def topological_scores(
        self,
        topology: Dict[str, object],
        baseline_topology: Dict[str, object],
        service_metrics: Dict[str, float],
        energy_metrics: Dict[str, float],
    ) -> Tuple[float, float, float, float]:
        """
        Combine surrogate graph invariants, persistence-diagram distances, and
        higher-order Hodge diagnostics into one admissibility score.

        The scoring philosophy follows the article closely:
        structural realism is not identical to small perturbation magnitude.
        A scenario may be difficult yet still admissible if its cross-domain
        structure remains coherent.
        """
        base_similarity, base_invariants, _, _ = AutonomousUrbanScenarioSimulator.topological_scores(
            self,
            topology=topology["curve"],
            baseline_topology=baseline_topology["curve"],
            service_metrics=service_metrics,
            energy_metrics=energy_metrics,
        )

        zone_h0_distance = self.safe_persim_distance(topology["zone_diagrams"][0], baseline_topology["zone_diagrams"][0])
        zone_h1_distance = self.safe_persim_distance(topology["zone_diagrams"][1], baseline_topology["zone_diagrams"][1])
        energy_h0_distance = self.safe_persim_distance(
            topology["energy_diagrams"][0],
            baseline_topology["energy_diagrams"][0],
        )

        entropy_shift = abs(float(topology["zone_entropy_h1"]) - float(baseline_topology["zone_entropy_h1"]))
        harmonic_shift = abs(float(topology["harmonic_count"]) - float(baseline_topology["harmonic_count"]))
        harmonic_shift = harmonic_shift / max(float(baseline_topology["harmonic_count"]), 1.0)

        spectral_shift = abs(float(topology["spectral_gap"]) - float(baseline_topology["spectral_gap"]))
        spectral_shift = spectral_shift / max(abs(float(baseline_topology["spectral_gap"])), 1e-6)

        triangle_baseline = max(float(baseline_topology["triangle_count"]), 1.0)
        triangle_retention = min(float(topology["triangle_count"]), triangle_baseline) / triangle_baseline

        diagram_distance = (
            0.25 * zone_h0_distance
            + 0.45 * zone_h1_distance
            + 0.20 * energy_h0_distance
            + 0.10 * entropy_shift
        )
        diagram_similarity = float(np.exp(-diagram_distance))
        higher_order_similarity = float(np.exp(-(0.60 * harmonic_shift + 0.40 * spectral_shift)))

        invariants = float(
            0.30 * base_invariants
            + 0.20 * float(service_metrics["reachability_ratio"])
            + 0.20 * float(energy_metrics["connected_load_ratio"])
            + 0.15 * float(topology["gcc_ratio_hodge"])
            + 0.15 * triangle_retention
        )

        similarity = float(0.35 * base_similarity + 0.40 * diagram_similarity + 0.25 * higher_order_similarity)
        topology_validity = float(np.clip(0.55 * similarity + 0.45 * invariants, 0.0, 1.0))
        boundary_score = float(np.exp(-abs(topology_validity - 0.84) / 0.055))

        return similarity, invariants, topology_validity, boundary_score

    # ------------------------------------------------------------------
    # Scenario summarization with explicit TDA metrics
    # ------------------------------------------------------------------
    def summarize_scenario(
        self,
        program: ScenarioProgram,
        controller_name: str = "Incumbent",
        learned_assets: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """
        Aggregate one scenario rollout into the metrics needed by the chapter.

        The summary mixes domain burdens and topology-aware descriptors so that:
        - the scenario bank can be curated,
        - difficulty can be compared against realism,
        - downstream controller analysis remains tied to the chapter objective.
        """
        timeseries, peak_index = self.simulate_scenario(
            program=program,
            controller_name=controller_name,
            learned_assets=learned_assets,
        )
        peak_row = timeseries[peak_index]
        topology_report = self.topology_report_from_peak_row(peak_row)

        topology_similarity, topology_invariants, topology_validity, boundary_score = self.topological_scores(
            topology=topology_report,
            baseline_topology=self.baseline_summary["topology"],
            service_metrics={"reachability_ratio": peak_row["reachability_ratio"]},
            energy_metrics={"connected_load_ratio": peak_row["connected_load_ratio"]},
        )

        travel_ratio = float(np.mean([row["mobility_mean_tt"] for row in timeseries]) / self.baseline_summary["mobility_mean_tt"])
        wait_ratio = float(np.mean([row["mean_wait"] for row in timeseries]) / self.baseline_summary["mean_wait"])
        overload_mean = float(np.mean([row["mean_overload"] for row in timeseries]))
        access_dispersion = float(np.mean([row["access_dispersion"] for row in timeseries]))
        access_dispersion_ratio = float(access_dispersion / self.baseline_summary["access_dispersion"])
        energy_score = float(overload_mean / 0.04)

        zone_h0_distance = self.safe_persim_distance(
            topology_report["zone_diagrams"][0],
            self.baseline_summary["topology"]["zone_diagrams"][0],
        )
        zone_h1_distance = self.safe_persim_distance(
            topology_report["zone_diagrams"][1],
            self.baseline_summary["topology"]["zone_diagrams"][1],
        )
        energy_h0_distance = self.safe_persim_distance(
            topology_report["energy_diagrams"][0],
            self.baseline_summary["topology"]["energy_diagrams"][0],
        )

        loss = float(
            0.40 * (travel_ratio - 1.0)
            + 0.18 * energy_score
            + 0.24 * (wait_ratio - 1.0)
            + 0.08 * (access_dispersion_ratio - 1.0)
            + 0.10 * zone_h1_distance
        )

        feasible = int(
            peak_row["reachability_ratio"] >= 0.65
            and np.max([row["mean_overload"] for row in timeseries]) < 0.90
            and topology_validity >= 0.72
        )

        return {
            "family": program.family,
            "intensity": float(program.intensity),
            "travel_ratio": travel_ratio,
            "wait_ratio": wait_ratio,
            "energy_score": energy_score,
            "access_disp_ratio": access_dispersion_ratio,
            "loss": loss,
            "difficulty": loss,
            "topology_similarity": topology_similarity,
            "topology_invariants": topology_invariants,
            "topology_validity": topology_validity,
            "boundary_score": boundary_score,
            "reachability": float(peak_row["reachability_ratio"]),
            "overload_peak": float(np.max([row["mean_overload"] for row in timeseries])),
            "feasible": feasible,
            "peak_time": int(peak_row["t"]),
            "zone_h0_distance": float(zone_h0_distance),
            "zone_h1_distance": float(zone_h1_distance),
            "energy_h0_distance": float(energy_h0_distance),
            "zone_entropy_h1": float(topology_report["zone_entropy_h1"]),
            "triangle_count": float(topology_report["triangle_count"]),
            "harmonic_count": float(topology_report["harmonic_count"]),
            "spectral_gap": float(topology_report["spectral_gap"]),
            "topology_report": topology_report,
            "timeseries": timeseries,
        }

    def scenario_fingerprint(self, program: ScenarioProgram, summary: Dict[str, object]) -> np.ndarray:
        """
        Enrich the base fingerprint with explicit algebraic-topology features.

        The extended fingerprint is used by:
        - novelty search,
        - scenario-bank embedding,
        - drift-aware coverage diagnostics,
        - Mapper visualization.
        """
        base = AutonomousUrbanScenarioSimulator.scenario_fingerprint(self, program, summary)
        extra = np.array(
            [
                float(summary["zone_h0_distance"]),
                float(summary["zone_h1_distance"]),
                float(summary["energy_h0_distance"]),
                float(summary["zone_entropy_h1"]),
                float(summary["harmonic_count"] / max(summary["triangle_count"], 1.0)),
                float(summary["spectral_gap"]),
            ],
            dtype=float,
        )
        return np.concatenate([base, extra])

    # ------------------------------------------------------------------
    # Scenario-bank learning with torch embeddings
    # ------------------------------------------------------------------
    def learn_assets_from_bank(self, bank: Sequence[Dict[str, object]]) -> Dict[str, object]:
        """
        Learn recurrent critical assets and a nonlinear scenario-bank embedding.

        The critical-asset logic is inherited from the original simulator, while
        the latent embedding is added so that the bank becomes a searchable,
        compact structural memory.
        """
        learned_assets = AutonomousUrbanScenarioSimulator.learn_assets_from_bank(self, bank)

        if len(bank) == 0:
            learned_assets.update(
                {
                    "fingerprint_scaler": None,
                    "embedding_model": None,
                    "latent_embeddings": np.zeros((0, 2), dtype=float),
                    "family_centroids": {},
                }
            )
            return learned_assets

        fingerprint_matrix = np.vstack(
            [self.scenario_fingerprint(item["program"], item["summary"]) for item in bank]
        )
        scaler = StandardScaler().fit(fingerprint_matrix)
        x_scaled = scaler.transform(fingerprint_matrix)

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        latent_dim = 4 if x_scaled.shape[1] >= 6 else max(2, x_scaled.shape[1] - 1)
        model = ScenarioEmbeddingAutoencoder(input_dim=x_scaled.shape[1], latent_dim=latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

        topo_idx = [
            x_scaled.shape[1] - 6,
            x_scaled.shape[1] - 5,
            x_scaled.shape[1] - 4,
            x_scaled.shape[1] - 3,
            x_scaled.shape[1] - 2,
            x_scaled.shape[1] - 1,
        ]

        model.train()
        for _ in range(260):
            optimizer.zero_grad()
            reconstruction, z_tensor = model(x_tensor)
            reconstruction_loss = torch.mean((reconstruction - x_tensor) ** 2)

            topo_tensor = x_tensor[:, topo_idx]
            target_distances = torch.cdist(topo_tensor, topo_tensor, p=2)
            latent_distances = torch.cdist(z_tensor, z_tensor, p=2)
            topology_regularizer = torch.mean((latent_distances - target_distances) ** 2)

            loss = reconstruction_loss + 0.05 * topology_regularizer
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            latent_embeddings = model.encode(x_tensor).cpu().numpy()

        family_centroids: Dict[str, np.ndarray] = {}
        for family in self.families:
            family_indices = [idx for idx, item in enumerate(bank) if item["program"].family == family]
            if family_indices:
                family_centroids[family] = np.mean(latent_embeddings[family_indices], axis=0)

        learned_assets.update(
            {
                "fingerprint_scaler": scaler,
                "embedding_model": model,
                "latent_embeddings": latent_embeddings,
                "family_centroids": family_centroids,
            }
        )
        return learned_assets

    @staticmethod
    def _as_2d_array(values: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Normalize arbitrary numeric inputs to a dense two-dimensional array.

        Several later plotting and encoding stages may receive a single vector,
        an empty collection, or a full matrix. This helper keeps all those
        cases shape-safe.
        """
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return np.zeros((0, 0), dtype=float)
        if array.ndim == 1:
            return array.reshape(1, -1)
        return array

    def safe_two_dimensional_projection(
        self,
        matrix: Sequence[np.ndarray] | np.ndarray,
        seed_offset: int = 0,
    ) -> np.ndarray:
        """
        Project embeddings to two coordinates without failing on tiny banks.

        PCA is used when it is mathematically valid. If the input is degenerate
        (for example a single scenario or a single feature), the method pads the
        missing axis with zeros instead of raising an exception.
        """
        matrix_2d = self._as_2d_array(matrix)
        if matrix_2d.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)
        if matrix_2d.shape[1] == 0:
            return np.zeros((matrix_2d.shape[0], 2), dtype=float)
        if matrix_2d.shape[0] == 1:
            if matrix_2d.shape[1] >= 2:
                return matrix_2d[:, :2].copy()
            return np.hstack([matrix_2d, np.zeros((1, 1), dtype=float)])
        if matrix_2d.shape[1] == 1:
            return np.hstack([matrix_2d, np.zeros((matrix_2d.shape[0], 1), dtype=float)])

        n_components = min(2, matrix_2d.shape[0], matrix_2d.shape[1])
        projected = PCA(
            n_components=n_components,
            random_state=self.master_seed + seed_offset,
        ).fit_transform(matrix_2d)

        if projected.shape[1] == 1:
            projected = np.hstack([projected, np.zeros((projected.shape[0], 1), dtype=float)])
        return projected[:, :2]

    def project_embedding_sets(
        self,
        named_sets: Dict[str, Sequence[np.ndarray] | np.ndarray],
        seed_offset: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Jointly project multiple embedding sets into one shared 2D frame.

        This guarantees that the original bank, hold-out bank, and refreshed
        bank are all visualized in comparable coordinates.
        """
        normalized_sets: Dict[str, np.ndarray] = {
            name: self._as_2d_array(values)
            for name, values in named_sets.items()
        }
        non_empty_items = [
            (name, values)
            for name, values in normalized_sets.items()
            if values.shape[0] > 0
        ]
        if not non_empty_items:
            return {name: np.zeros((0, 2), dtype=float) for name in named_sets}

        stacked = np.vstack([values for _, values in non_empty_items])
        projected_stacked = self.safe_two_dimensional_projection(
            stacked,
            seed_offset=seed_offset,
        )

        projected_sets: Dict[str, np.ndarray] = {}
        cursor = 0
        for name, values in normalized_sets.items():
            count = values.shape[0]
            if count == 0:
                projected_sets[name] = np.zeros((0, 2), dtype=float)
            else:
                projected_sets[name] = projected_stacked[cursor : cursor + count]
                cursor += count

        return projected_sets

    def encode_fingerprints(
        self,
        fingerprints: Sequence[np.ndarray] | np.ndarray,
        learned_assets: Dict[str, object],
    ) -> np.ndarray:
        """
        Encode one or many fingerprints into the latent scenario-memory space.
        """
        model = learned_assets.get("embedding_model")
        scaler = learned_assets.get("fingerprint_scaler")

        if isinstance(fingerprints, np.ndarray):
            matrix = self._as_2d_array(fingerprints)
        else:
            fingerprint_list = [np.asarray(fingerprint, dtype=float) for fingerprint in fingerprints]
            if len(fingerprint_list) == 0:
                if model is not None and hasattr(model, "encoder") and len(model.encoder) > 0 and hasattr(model.encoder[-1], "out_features"):
                    return np.zeros((0, int(model.encoder[-1].out_features)), dtype=float)
                if scaler is not None and hasattr(scaler, "mean_"):
                    return np.zeros((0, len(scaler.mean_)), dtype=float)
                return np.zeros((0, 0), dtype=float)
            matrix = np.vstack(fingerprint_list)

        if matrix.shape[0] == 0:
            if model is not None and hasattr(model, "encoder") and len(model.encoder) > 0 and hasattr(model.encoder[-1], "out_features"):
                return np.zeros((0, int(model.encoder[-1].out_features)), dtype=float)
            return matrix

        if scaler is not None:
            matrix = scaler.transform(matrix)

        if model is None:
            return matrix

        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(matrix, dtype=torch.float32)
            return model.encode(tensor).cpu().numpy()

    # ------------------------------------------------------------------
    # Drift monitoring: add latent payloads for section 5.3 figures
    # ------------------------------------------------------------------
    def evaluate_drift_epoch(
        self,
        epoch_name: str,
        original_bank: Sequence[Dict[str, object]],
        original_fingerprints: Sequence[np.ndarray],
        master_baseline_topology: Dict[str, object],
        seed: int,
        original_learned_assets: Dict[str, object],
    ) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
        """
        Evaluate one operating epoch under structural drift and also return the
        latent embeddings needed by the Section 5.3 visualization.
        """
        self.restore_original_state()
        self.apply_drift(epoch_name=epoch_name)

        similarity, _, _, _ = self.topological_scores(
            topology=self.baseline_summary["topology"],
            baseline_topology=master_baseline_topology,
            service_metrics={"reachability_ratio": 1.0},
            energy_metrics={"connected_load_ratio": 1.0},
        )
        drift_score = float(1.0 - similarity)

        refresh_bank, _ = self.generate_scenario_bank(
            num_generations=4,
            population_size=24,
            bank_size=14,
            seed=seed,
        )

        holdout_bank = self.mutate_bank(
            bank=refresh_bank,
            seed=seed + 999,
            target_size=12,
        )

        refresh_bank_fingerprints = self.bank_fingerprints(refresh_bank)
        holdout_with_fingerprints = []
        for item in holdout_bank:
            holdout_with_fingerprints.append(
                {
                    "program": item["program"],
                    "summary": item["summary"],
                    "fingerprint": self.scenario_fingerprint(item["program"], item["summary"]),
                }
            )

        coverage_before = self.coverage_metrics(
            target_bank=holdout_with_fingerprints,
            reference_fingerprints=original_fingerprints,
            threshold=0.40,
        )
        coverage_after = self.coverage_metrics(
            target_bank=holdout_with_fingerprints,
            reference_fingerprints=list(original_fingerprints) + refresh_bank_fingerprints,
            threshold=0.40,
        )

        family_counter = Counter(item["program"].family for item in refresh_bank)
        dominant_family = family_counter.most_common(1)[0][0] if family_counter else ""

        refresh_embeddings = self.encode_fingerprints(refresh_bank_fingerprints, original_learned_assets)
        holdout_embeddings = self.encode_fingerprints(
            [item["fingerprint"] for item in holdout_with_fingerprints],
            original_learned_assets,
        )

        payload = {
            "refresh_embeddings": refresh_embeddings,
            "holdout_embeddings": holdout_embeddings,
        }

        distance_before = float(coverage_before["mean_nearest_distance"])
        distance_after = float(coverage_after["mean_nearest_distance"])
        if not np.isfinite(distance_before) or distance_before <= 1e-12 or not np.isfinite(distance_after):
            distance_reduction = 0.0
        else:
            distance_reduction = float(1.0 - distance_after / distance_before)

        row = {
            "epoch": epoch_name,
            "drift_score": drift_score,
            "dominant_family": dominant_family,
            "coverage_before": float(coverage_before["coverage"]),
            "coverage_after": float(coverage_after["coverage"]),
            "distance_before": distance_before,
            "distance_after": distance_after,
            "coverage_lift": float(coverage_after["coverage"] - coverage_before["coverage"]),
            "distance_reduction": distance_reduction,
            "refresh_bank_size": len(refresh_bank),
            "holdout_size": len(holdout_with_fingerprints),
        }
        return row, payload

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def build_table_1(self, bank: Sequence[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Section 5.1 table with explicit algebraic-topology diagnostics.
        """
        records = []
        for index, item in enumerate(bank, start=1):
            program = item["program"]
            summary = item["summary"]
            records.append(
                {
                    "scenario_id": index,
                    "family": program.family,
                    "intensity": float(program.intensity),
                    "difficulty": float(summary["difficulty"]),
                    "topology_validity": float(summary["topology_validity"]),
                    "boundary_score": float(summary["boundary_score"]),
                    "novelty": float(item["novelty"]),
                    "travel_ratio": float(summary["travel_ratio"]),
                    "wait_ratio": float(summary["wait_ratio"]),
                    "energy_score": float(summary["energy_score"]),
                    "overload_peak": float(summary["overload_peak"]),
                    "zone_h1_distance": float(summary["zone_h1_distance"]),
                    "energy_h0_distance": float(summary["energy_h0_distance"]),
                    "harmonic_count": float(summary["harmonic_count"]),
                }
            )

        bank_df = pd.DataFrame(records)
        table_1 = (
            bank_df.groupby("family")
            .agg(
                accepted=("scenario_id", "count"),
                difficulty=("difficulty", "mean"),
                topology_validity=("topology_validity", "mean"),
                boundary_score=("boundary_score", "mean"),
                travel_ratio=("travel_ratio", "mean"),
                wait_ratio=("wait_ratio", "mean"),
                energy_score=("energy_score", "mean"),
                peak_overload=("overload_peak", "mean"),
                zone_h1_distance=("zone_h1_distance", "mean"),
                harmonic_count=("harmonic_count", "mean"),
            )
            .reset_index()
            .sort_values("family")
        )
        return table_1, bank_df

    # ------------------------------------------------------------------
    # Mapper and persistence figures for Section 5.1
    # ------------------------------------------------------------------
    def mapper_graph(self, bank_df: pd.DataFrame, learned_assets: Dict[str, object]) -> Dict[str, object]:
        """
        Build a Mapper graph over the latent scenario-bank embedding.

        The method first standardizes the latent coordinates and then tries a
        small grid of cover/DBSCAN settings. This avoids the common failure mode
        where KeplerMapper returns an empty graph for a perfectly valid but
        sparse bank.
        """
        raw_embeddings = self._as_2d_array(
            np.asarray(learned_assets.get("latent_embeddings", np.zeros((0, 0), dtype=float)), dtype=float)
        )
        if raw_embeddings.shape[0] > 1 and raw_embeddings.shape[1] > 0:
            latent_embeddings = StandardScaler().fit_transform(raw_embeddings)
        else:
            latent_embeddings = raw_embeddings.copy()

        lens = self.safe_two_dimensional_projection(latent_embeddings, seed_offset=51)
        mapper = km.KeplerMapper(verbose=0)

        parameter_grid = [
            {"n_cubes": 6, "perc_overlap": 0.35, "eps": 0.75, "min_samples": 2},
            {"n_cubes": 5, "perc_overlap": 0.40, "eps": 0.95, "min_samples": 2},
            {"n_cubes": 4, "perc_overlap": 0.45, "eps": 1.15, "min_samples": 2},
            {"n_cubes": 4, "perc_overlap": 0.50, "eps": 1.35, "min_samples": 1},
        ]

        graph: Dict[str, object] = {"nodes": {}, "links": {}, "simplices": []}
        selected_config: Optional[Dict[str, float]] = None
        fallback_reason = ""
        last_error: Optional[Exception] = None

        if latent_embeddings.shape[0] > 0:
            for config in parameter_grid:
                try:
                    cover = km.Cover(
                        n_cubes=int(config["n_cubes"]),
                        perc_overlap=float(config["perc_overlap"]),
                    )
                    clusterer = DBSCAN(
                        eps=float(config["eps"]),
                        min_samples=int(config["min_samples"]),
                    )
                    candidate_graph = mapper.map(
                        lens,
                        latent_embeddings,
                        cover=cover,
                        clusterer=clusterer,
                    )
                    if len(candidate_graph.get("nodes", {})) > 0:
                        graph = candidate_graph
                        selected_config = config
                        break
                    fallback_reason = (
                        "KeplerMapper produced zero nodes for all tested cover/DBSCAN "
                        "settings, so the pipeline will export a latent-projection fallback."
                    )
                except Exception as exc:
                    last_error = exc
                    fallback_reason = (
                        "KeplerMapper failed while building the Mapper graph, so the "
                        "pipeline will export a latent-projection fallback."
                    )

        if selected_config is None and last_error is not None:
            fallback_reason = f"{fallback_reason} Last error: {last_error}"

        node_families = {}
        for node_name, members in graph.get("nodes", {}).items():
            member_indices = [int(member) for member in members if int(member) < len(bank_df)]
            if not member_indices:
                continue
            member_families = bank_df.iloc[member_indices]["family"].tolist()
            if member_families:
                node_families[node_name] = Counter(member_families).most_common(1)[0][0]

        return {
            "graph": graph,
            "lens": lens,
            "node_families": node_families,
            "has_nodes": bool(graph.get("nodes")),
            "mapper_config": selected_config,
            "fallback_reason": fallback_reason,
        }

    def write_latent_projection_html(
        self,
        html_path: str,
        bank_df: pd.DataFrame,
        projection: np.ndarray,
        title: str,
        subtitle: str,
        family_palette: Dict[str, str],
    ) -> None:
        """
        Write a self-contained HTML fallback when Mapper visualization is empty.

        The fallback keeps the pipeline functional and still provides an
        interpretable view of the latent scenario geometry.
        """
        output_parent = os.path.dirname(html_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)

        projection_2d = self._as_2d_array(projection)
        if projection_2d.shape[1] < 2:
            projection_2d = self.safe_two_dimensional_projection(projection_2d, seed_offset=151)

        n_points = min(len(bank_df), projection_2d.shape[0])

        if n_points == 0:
            html_string = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f1f1f; }}
.card {{ max-width: 980px; margin: 0 auto; padding: 24px; border: 1px solid #d9d9d9; border-radius: 12px; }}
h1 {{ margin-top: 0; color: #17365d; }}
p {{ line-height: 1.5; }}
</style>
</head>
<body>
<div class="card">
<h1>{html.escape(title)}</h1>
<p>{html.escape(subtitle)}</p>
<p>No latent points were available to render.</p>
</div>
</body>
</html>"""
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(html_string)
            return

        display_df = bank_df.iloc[:n_points].copy().reset_index(drop=True)
        x_values = projection_2d[:n_points, 0]
        y_values = projection_2d[:n_points, 1]

        def scale_axis(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
            value_min = float(np.min(values))
            value_max = float(np.max(values))
            if abs(value_max - value_min) < 1e-12:
                return np.full(values.shape, (lower + upper) / 2.0, dtype=float)
            return lower + (values - value_min) * (upper - lower) / (value_max - value_min)

        svg_x = scale_axis(x_values, 70.0, 790.0)
        svg_y = scale_axis(-y_values, 70.0, 450.0)

        has_scenario_id = "scenario_id" in display_df.columns
        circles_html: List[str] = []
        for index, row in display_df.iterrows():
            scenario_label = int(row["scenario_id"]) if has_scenario_id else index + 1
            family = str(row["family"])
            color = family_palette.get(family, "#4f81bd")
            difficulty = float(row["difficulty"]) if "difficulty" in display_df.columns else float("nan")
            topology_validity = (
                float(row["topology_validity"])
                if "topology_validity" in display_df.columns
                else float("nan")
            )
            tooltip = html.escape(
                f"Scenario {scenario_label} | Family: {family} | Difficulty: {difficulty:.3f} | "
                f"Topology validity: {topology_validity:.3f}"
            )
            circles_html.append(
                f'<circle cx="{svg_x[index]:.2f}" cy="{svg_y[index]:.2f}" r="7.5" '
                f'fill="{color}" fill-opacity="0.88" stroke="#ffffff" stroke-width="1.2">'
                f"<title>{tooltip}</title></circle>"
            )

        legend_items: List[str] = []
        for family in sorted(display_df["family"].unique()):
            color = family_palette.get(str(family), "#4f81bd")
            legend_items.append(
                f'<span class="legend-item"><span class="swatch" style="background:{color};"></span>'
                f"{html.escape(str(family))}</span>"
            )

        table_columns = [
            column
            for column in ["scenario_id", "family", "difficulty", "topology_validity", "boundary_score"]
            if column in display_df.columns
        ]
        table_html = display_df[table_columns].to_html(
            index=False,
            border=0,
            classes="scenario-table",
            float_format=lambda value: f"{value:.3f}",
        )

        html_string = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f1f1f; background: #fafafa; }}
.card {{ max-width: 1040px; margin: 0 auto; background: white; padding: 24px; border: 1px solid #d9d9d9; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }}
h1 {{ margin-top: 0; color: #17365d; }}
p {{ line-height: 1.55; }}
.figure-wrap {{ margin-top: 18px; }}
svg {{ width: 100%; height: auto; border: 1px solid #d9e2f3; border-radius: 10px; background: #fcfdff; }}
.axis-label {{ font-size: 13px; fill: #4d4d4d; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 16px 0 12px; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 8px; font-size: 13px; }}
.swatch {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
.note {{ margin-top: 10px; padding: 12px 14px; background: #f6f8fb; border-left: 4px solid #5b9bd5; }}
table.scenario-table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 13px; }}
table.scenario-table th, table.scenario-table td {{ border: 1px solid #d9d9d9; padding: 8px 10px; text-align: left; }}
table.scenario-table th {{ background: #f2f6fc; }}
</style>
</head>
<body>
<div class="card">
<h1>{html.escape(title)}</h1>
<p>{html.escape(subtitle)}</p>

<div class="figure-wrap">
<svg viewBox="0 0 860 520" role="img" aria-label="Latent projection of the accepted scenario bank">
<rect x="0" y="0" width="860" height="520" fill="#ffffff"></rect>
<line x1="60" y1="460" x2="810" y2="460" stroke="#7f7f7f" stroke-width="1.4"></line>
<line x1="60" y1="460" x2="60" y2="60" stroke="#7f7f7f" stroke-width="1.4"></line>
<text x="435" y="500" text-anchor="middle" class="axis-label">Latent axis 1</text>
<text x="20" y="260" text-anchor="middle" class="axis-label" transform="rotate(-90 20 260)">Latent axis 2</text>
{''.join(circles_html)}
</svg>
</div>

<div class="legend">
{''.join(legend_items)}
</div>

<div class="note">
Mapper produced zero nodes under the tested parameters, so this fallback view preserves the latent scenario geometry and keeps the export pipeline complete.
</div>

{table_html}
</div>
</body>
</html>"""

        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(html_string)

    def plot_section_5_1(
        self,
        bank_df: pd.DataFrame,
        bank: Sequence[Dict[str, object]],
        learned_assets: Dict[str, object],
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Create all Section 5.1 figures.

        In addition to the core difficulty-validity and burden figures, this
        method exports:
        - an overlay of baseline vs hardest-scenario persistence diagrams,
        - a static Mapper graph when available,
        - a robust fallback latent projection when Mapper becomes empty.
        """
        os.makedirs(output_dir, exist_ok=True)
        family_palette = {
            "ClinicOutage": "#1f77b4",
            "CorridorIncident": "#ff7f0e",
            "EVChargingShock": "#2ca02c",
            "HeatwaveLoad": "#d62728",
            "StadiumEvent": "#9467bd",
            "StormCascade": "#8c564b",
        }

        # Figure 5.1-A: difficulty versus topology validity.
        fig, axis = plt.subplots(figsize=(8.6, 5.6))
        for family in sorted(bank_df["family"].unique()):
            subset = bank_df[bank_df["family"] == family]
            axis.scatter(
                subset["topology_validity"],
                subset["difficulty"],
                s=120 + 220 * subset["boundary_score"],
                alpha=0.82,
                label=family,
                c=family_palette[family],
                edgecolor="white",
                linewidth=0.9,
            )

        axis.axvspan(0.78, 0.86, color="#f8d5a8", alpha=0.35, label="Boundary regime")
        axis.axvline(0.78, color="#bb7722", linestyle="--", linewidth=1.2)
        axis.set_xlabel("Topology validity score")
        axis.set_ylabel("Scenario difficulty")
        axis.set_title("Accepted scenarios: difficulty versus structural admissibility")
        axis.grid(True, alpha=0.18)
        axis.legend(ncols=2, fontsize=8, frameon=False, loc="upper left")
        fig.tight_layout()
        path_a = os.path.join(output_dir, "figure_5_1_A_difficulty_vs_topology.png")
        fig.savefig(path_a, dpi=260, bbox_inches="tight")
        plt.close(fig)

        # Figure 5.1-B: burden heatmap.
        heat_df = bank_df.groupby("family")[["travel_ratio", "wait_ratio", "energy_score", "topology_validity"]].mean().copy()
        heat_df["travel_burden"] = heat_df["travel_ratio"] - 1.0
        heat_df["service_burden"] = heat_df["wait_ratio"] - 1.0
        heat_df["topology_deviation"] = 1.0 - heat_df["topology_validity"]
        heat_values = heat_df[["travel_burden", "service_burden", "energy_score", "topology_deviation"]]
        heat_normalized = (heat_values - heat_values.min()) / (heat_values.max() - heat_values.min() + 1e-9)

        fig, axis = plt.subplots(figsize=(8.6, 4.8))
        image = axis.imshow(heat_normalized.values, aspect="auto", cmap="YlOrRd")
        axis.set_yticks(range(len(heat_normalized.index)))
        axis.set_yticklabels(heat_normalized.index)
        axis.set_xticks(range(len(heat_normalized.columns)))
        axis.set_xticklabels(["Travel\nburden", "Service\nburden", "Energy\nstress", "Topology\ndeviation"])

        for i_row in range(heat_normalized.shape[0]):
            for j_col in range(heat_normalized.shape[1]):
                axis.text(j_col, i_row, f"{heat_values.iloc[i_row, j_col]:.2f}", ha="center", va="center", fontsize=8)

        axis.set_title("Mean normalized burden profile by accepted scenario family")
        colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
        colorbar.set_label("Column-wise normalized intensity")
        fig.tight_layout()
        path_b = os.path.join(output_dir, "figure_5_1_B_burden_heatmap.png")
        fig.savefig(path_b, dpi=260, bbox_inches="tight")
        plt.close(fig)

        # Figure 5.1-C: baseline versus hardest scenario persistence diagram.
        hardest_item = max(bank, key=lambda item: float(item["summary"]["difficulty"]))
        baseline_h1 = diagram_with_fallback(self.baseline_summary["topology"]["zone_diagrams"][1])
        hardest_h1 = diagram_with_fallback(hardest_item["summary"]["topology_report"]["zone_diagrams"][1])

        fig, axis = plt.subplots(figsize=(6.8, 5.4))
        plot_diagrams(
            [baseline_h1, hardest_h1],
            labels=["Baseline H1", f'Hardest H1: {hardest_item["program"].family}'],
            title="Persistent H1 comparison: baseline vs hardest accepted scenario",
            ax=axis,
            legend=True,
            show=False,
        )
        path_c = os.path.join(output_dir, "figure_5_1_C_persistence_diagram_baseline_vs_hardest.png")
        fig.savefig(path_c, dpi=260, bbox_inches="tight")
        plt.close(fig)

        # Figure 5.1-D: Mapper graph or robust fallback projection.
        mapper_payload = self.mapper_graph(bank_df=bank_df, learned_assets=learned_assets)
        mapper_graph = mapper_payload["graph"]
        node_families = mapper_payload["node_families"]

        html_path = os.path.join(output_dir, "figure_5_1_D_mapper_graph_interactive.html")
        interactive_mapper_ok = False
        if mapper_payload["has_nodes"]:
            try:
                km.KeplerMapper(verbose=0).visualize(
                    mapper_graph,
                    path_html=html_path,
                    title="Accepted scenario bank - Mapper graph",
                )
                interactive_mapper_ok = True
            except Exception as exc:
                mapper_payload["fallback_reason"] = (
                    f"KeplerMapper.visualize failed after a non-empty graph was built. "
                    f"Fallback HTML was written instead. Error: {exc}"
                )

        if not interactive_mapper_ok:
            fallback_subtitle = mapper_payload["fallback_reason"] or (
                "Fallback projection of the accepted scenario bank in the learned latent space."
            )
            self.write_latent_projection_html(
                html_path=html_path,
                bank_df=bank_df,
                projection=mapper_payload["lens"],
                title="Accepted scenario bank - latent projection fallback",
                subtitle=fallback_subtitle,
                family_palette=family_palette,
            )

        fig, axis = plt.subplots(figsize=(8.0, 6.2))
        if mapper_payload["has_nodes"]:
            static_graph = nx.Graph()
            for node_name, members in mapper_graph["nodes"].items():
                static_graph.add_node(
                    node_name,
                    size=max(len(members), 1),
                    family=node_families.get(node_name, "Unknown"),
                )

            for source_node, target_nodes in mapper_graph["links"].items():
                for target_node in target_nodes:
                    if source_node != target_node:
                        static_graph.add_edge(source_node, target_node)

            if static_graph.number_of_nodes() > 0:
                layout = nx.spring_layout(static_graph, seed=self.master_seed, k=0.8)
                nx.draw_networkx_edges(static_graph, layout, alpha=0.35, width=1.2, ax=axis)

                for family, color in family_palette.items():
                    node_list = [node for node, data in static_graph.nodes(data=True) if data["family"] == family]
                    if not node_list:
                        continue
                    sizes = [140 + 55 * static_graph.nodes[node]["size"] for node in node_list]
                    nx.draw_networkx_nodes(
                        static_graph,
                        layout,
                        nodelist=node_list,
                        node_size=sizes,
                        node_color=color,
                        alpha=0.88,
                        edgecolors="white",
                        linewidths=0.7,
                        ax=axis,
                        label=family,
                    )

                axis.set_title("Mapper graph of the accepted scenario bank")
                axis.axis("off")
            else:
                mapper_payload["has_nodes"] = False

        if not mapper_payload["has_nodes"]:
            projection = mapper_payload["lens"]
            projected_df = bank_df.iloc[: projection.shape[0]].copy().reset_index(drop=True)

            if projection.shape[0] == 0:
                axis.text(
                    0.5,
                    0.5,
                    "No latent embeddings were available for the fallback view.",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                axis.axis("off")
            else:
                for family in sorted(projected_df["family"].unique()):
                    mask = (projected_df["family"] == family).to_numpy()
                    axis.scatter(
                        projection[mask, 0],
                        projection[mask, 1],
                        s=88,
                        alpha=0.84,
                        label=family,
                        c=family_palette[family],
                        edgecolor="white",
                        linewidth=0.8,
                    )

                if "scenario_id" in projected_df.columns:
                    for row_index, scenario_id in enumerate(projected_df["scenario_id"].tolist()):
                        axis.text(
                            projection[row_index, 0] + 0.02,
                            projection[row_index, 1] + 0.02,
                            str(int(scenario_id)),
                            fontsize=7,
                            alpha=0.75,
                        )

                axis.set_xlabel("Latent axis 1")
                axis.set_ylabel("Latent axis 2")
                axis.set_title("Latent scenario-bank projection (fallback when Mapper yields zero nodes)")
                axis.grid(True, alpha=0.18)

        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False, fontsize=8, ncols=2, loc="upper center")

        path_d = os.path.join(output_dir, "figure_5_1_D_mapper_graph_static.png")
        fig.tight_layout()
        fig.savefig(path_d, dpi=260, bbox_inches="tight")
        plt.close(fig)

        return {
            "figure_5_1_A": path_a,
            "figure_5_1_B": path_b,
            "figure_5_1_C": path_c,
            "figure_5_1_D_static": path_d,
            "figure_5_1_D_html": html_path,
        }

    # ------------------------------------------------------------------
    # Section 5.2 figures
    # ------------------------------------------------------------------
    def plot_section_5_2(
        self,
        controller_df: pd.DataFrame,
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Create the Section 5.2 figures.

        The first two figures mirror the chapter narrative directly. A third
        figure is added to highlight controller gains by scenario family.
        Missing families are represented explicitly as unavailable rather than
        causing a hard failure.
        """
        os.makedirs(output_dir, exist_ok=True)
        family_controller = ordered_controller_family_loss_matrix(controller_df)

        fig, axis = plt.subplots(figsize=(7.8, 4.8))
        image = nan_aware_imshow(axis, family_controller, cmap_name="Blues")
        axis.set_yticks(range(len(family_controller.index)))
        axis.set_yticklabels(family_controller.index)
        axis.set_xticks(range(len(family_controller.columns)))
        axis.set_xticklabels(["Incumbent", "Reactive", "Topology-aware"])
        annotate_dataframe_cells(axis, family_controller, fontsize=8)

        axis.set_title("Mean policy loss by scenario family and controller")
        colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
        colorbar.set_label("Loss")
        fig.tight_layout()
        path_a = os.path.join(output_dir, "figure_5_2_A_controller_family_heatmap.png")
        fig.savefig(path_a, dpi=260, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7.8, 5.0))
        data = [
            controller_df[controller_df["controller"] == controller_name]["loss"].values
            for controller_name in CONTROLLER_ORDER
        ]
        boxplot = axis.boxplot(
            data,
            patch_artist=True,
            tick_labels=["Incumbent", "Reactive", "Topology-aware"],
            medianprops=dict(color="black", linewidth=1.3),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        colors = ["#7ea6e0", "#8ac6a4", "#f3c178"]
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        for index, controller_name in enumerate(CONTROLLER_ORDER, start=1):
            values = controller_df[controller_df["controller"] == controller_name]["loss"].values
            x_values = np.random.default_rng(10 + index).normal(index, 0.04, size=len(values))
            axis.scatter(x_values, values, s=14, alpha=0.45, color="#444444")

        axis.set_ylabel("Scenario loss")
        axis.set_title("Distribution of policy losses across the accepted scenario bank")
        axis.grid(True, axis="y", alpha=0.18)
        fig.tight_layout()
        path_b = os.path.join(output_dir, "figure_5_2_B_loss_distribution.png")
        fig.savefig(path_b, dpi=260, bbox_inches="tight")
        plt.close(fig)

        gain_df = family_controller.copy()
        gain_df["TopologyAware_gain_vs_Incumbent"] = (
            (gain_df["Incumbent"] - gain_df["TopologyAware"]) / gain_df["Incumbent"].abs().replace(0.0, np.nan)
        )
        gain_values = 100.0 * gain_df["TopologyAware_gain_vs_Incumbent"].to_numpy(dtype=float)
        plotted_gain_values = np.nan_to_num(gain_values, nan=0.0)
        bar_colors = ["#5aa469" if np.isfinite(value) else "#d9d9d9" for value in gain_values]

        fig, axis = plt.subplots(figsize=(7.6, 4.8))
        axis.barh(
            gain_df.index,
            plotted_gain_values,
            color=bar_colors,
            alpha=0.90,
        )
        for y_index, value in enumerate(gain_values):
            if np.isfinite(value):
                x_position = value + 1.0 if value >= 0 else value - 6.0
                axis.text(x_position, y_index, f"{value:.1f}%", va="center", fontsize=8)
            else:
                axis.text(1.0, y_index, "NA", va="center", fontsize=8, color="#555555")
        axis.set_xlabel("Relative gain of topology-aware controller vs incumbent (%)")
        axis.set_title("Family-level robustness gains from topology-aware control")
        axis.grid(True, axis="x", alpha=0.18)
        fig.tight_layout()
        path_c = os.path.join(output_dir, "figure_5_2_C_relative_gain_by_family.png")
        fig.savefig(path_c, dpi=260, bbox_inches="tight")
        plt.close(fig)

        return {
            "figure_5_2_A": path_a,
            "figure_5_2_B": path_b,
            "figure_5_2_C": path_c,
        }

    # ------------------------------------------------------------------
    # Section 5.3 figures
    # ------------------------------------------------------------------
    def plot_section_5_3(
        self,
        drift_df: pd.DataFrame,
        latent_payloads: Dict[str, np.ndarray],
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Create the Section 5.3 figures, including a latent-space drift map.

        The latent-space plot is written defensively so that empty refresh or
        hold-out banks do not crash the export stage.
        """
        os.makedirs(output_dir, exist_ok=True)

        full_epochs = ["Baseline"] + drift_df["epoch"].tolist()
        drift_scores = [0.0] + drift_df["drift_score"].tolist()

        fig, axis = plt.subplots(figsize=(8.0, 4.6))
        x_values = np.arange(len(full_epochs))
        axis.plot(x_values, drift_scores, marker="o", linewidth=2.2, color="#2f6db3")
        for x_value, y_value in zip(x_values, drift_scores):
            axis.text(x_value, y_value + 0.006, f"{y_value:.2f}", ha="center", va="bottom", fontsize=8)
        axis.set_xticks(x_values)
        axis.set_xticklabels(["Baseline", "Construction", "EV adoption", "Service\ndecentralization"])
        axis.set_ylabel("Structural drift score")
        axis.set_title("Structural drift monitor across operating epochs")
        axis.grid(True, axis="y", alpha=0.18)
        fig.tight_layout()
        path_a = os.path.join(output_dir, "figure_5_3_A_structural_drift.png")
        fig.savefig(path_a, dpi=260, bbox_inches="tight")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(8.2, 4.8))
        x_positions = np.arange(len(drift_df))
        width = 0.34
        axis.bar(x_positions - width / 2, drift_df["coverage_before"], width=width, label="Before refresh", color="#b8cbe5")
        axis.bar(x_positions + width / 2, drift_df["coverage_after"], width=width, label="After refresh", color="#4f81bd")
        for x_value, before_value, after_value in zip(x_positions, drift_df["coverage_before"], drift_df["coverage_after"]):
            axis.text(x_value - width / 2, before_value + 0.015, f"{before_value:.2f}", ha="center", va="bottom", fontsize=8)
            axis.text(x_value + width / 2, after_value + 0.015, f"{after_value:.2f}", ha="center", va="bottom", fontsize=8)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(["Construction", "EV adoption", "Service\ndecentralization"])
        axis.set_ylabel("Hold-out coverage")
        axis.set_ylim(0.0, max(0.72, float(drift_df["coverage_after"].max()) + 0.10))
        axis.set_title("Scenario-bank coverage before and after refresh under drift")
        axis.legend(frameon=False)
        axis.grid(True, axis="y", alpha=0.18)
        fig.tight_layout()
        path_b = os.path.join(output_dir, "figure_5_3_B_coverage_refresh.png")
        fig.savefig(path_b, dpi=260, bbox_inches="tight")
        plt.close(fig)

        epoch_order = ["Construction season", "EV adoption surge", "Service decentralization"]
        embedding_sets: Dict[str, np.ndarray] = {
            "original": np.asarray(latent_payloads.get("original_embeddings", np.zeros((0, 0), dtype=float)), dtype=float),
        }
        for epoch in epoch_order:
            embedding_sets[f"{epoch}_holdout"] = np.asarray(
                latent_payloads.get(f"{epoch}_holdout", np.zeros((0, 0), dtype=float)),
                dtype=float,
            )
            embedding_sets[f"{epoch}_refresh"] = np.asarray(
                latent_payloads.get(f"{epoch}_refresh", np.zeros((0, 0), dtype=float)),
                dtype=float,
            )

        projected_sets = self.project_embedding_sets(
            embedding_sets,
            seed_offset=53,
        )

        fig, axis = plt.subplots(figsize=(8.2, 6.0))
        original_2d = projected_sets["original"]
        if original_2d.shape[0] > 0:
            axis.scatter(
                original_2d[:, 0],
                original_2d[:, 1],
                s=32,
                alpha=0.35,
                color="#999999",
                label="Original accepted bank",
            )

        epoch_colors = {
            "Construction season": "#d62728",
            "EV adoption surge": "#2ca02c",
            "Service decentralization": "#1f77b4",
        }

        plotted_any_epoch = bool(original_2d.shape[0] > 0)
        for epoch in epoch_order:
            color = epoch_colors[epoch]
            holdout_embeddings = projected_sets[f"{epoch}_holdout"]
            refresh_embeddings = projected_sets[f"{epoch}_refresh"]

            hold_centroid: Optional[np.ndarray] = None
            refresh_centroid: Optional[np.ndarray] = None

            if holdout_embeddings.shape[0] > 0:
                hold_centroid = np.mean(holdout_embeddings, axis=0)
                axis.scatter(
                    hold_centroid[0],
                    hold_centroid[1],
                    s=140,
                    marker="X",
                    color=color,
                    edgecolors="white",
                    linewidths=0.8,
                    label=f"{epoch} holdout",
                )
                plotted_any_epoch = True

            if refresh_embeddings.shape[0] > 0:
                refresh_centroid = np.mean(refresh_embeddings, axis=0)
                axis.scatter(
                    refresh_centroid[0],
                    refresh_centroid[1],
                    s=140,
                    marker="o",
                    facecolors="none",
                    edgecolors=color,
                    linewidths=2.0,
                    label=f"{epoch} refresh",
                )
                plotted_any_epoch = True

            if hold_centroid is not None and refresh_centroid is not None:
                axis.plot(
                    [hold_centroid[0], refresh_centroid[0]],
                    [hold_centroid[1], refresh_centroid[1]],
                    linestyle="--",
                    color=color,
                    alpha=0.8,
                )

        if not plotted_any_epoch:
            axis.text(
                0.5,
                0.5,
                "No latent embeddings were available for the drift map.",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            axis.axis("off")
        else:
            axis.set_xlabel("Latent axis 1")
            axis.set_ylabel("Latent axis 2")
            axis.set_title("Latent-space shift of drifted hold-out regimes and refreshed banks")
            axis.legend(frameon=False, fontsize=8, ncols=2, loc="best")
            axis.grid(True, alpha=0.18)

        fig.tight_layout()
        path_c = os.path.join(output_dir, "figure_5_3_C_latent_shift_refresh.png")
        fig.savefig(path_c, dpi=260, bbox_inches="tight")
        plt.close(fig)

        return {
            "figure_5_3_A": path_a,
            "figure_5_3_B": path_b,
            "figure_5_3_C": path_c,
        }

    # ------------------------------------------------------------------
    # Master execution method with section-aware export tree
    # ------------------------------------------------------------------
    def run_all(self, output_dir: str) -> Dict[str, str]:
        """
        Generate the complete chapter artifact package.

        This method is intentionally organized around the manuscript structure.
        The exported directory tree mirrors Sections 5.1, 5.2, and 5.3 exactly.
        """
        set_global_seed(self.master_seed)
        paths = ensure_output_tree(output_dir)

        self.restore_original_state()

        scenario_bank, generation_history = self.generate_scenario_bank(
            num_generations=6,
            population_size=28,
            bank_size=24,
            seed=self.master_seed,
        )

        learned_assets = self.learn_assets_from_bank(scenario_bank)
        controller_df, _ = self.evaluate_controllers(
            bank=scenario_bank,
            learned_assets=learned_assets,
        )

        table_1, bank_df = self.build_table_1(scenario_bank)
        table_2 = self.build_table_2(controller_df)

        master_baseline_topology = copy.deepcopy(self.baseline_summary["topology"])
        original_fingerprints = self.bank_fingerprints(scenario_bank)

        drift_rows = []
        latent_payloads = {
            "original_embeddings": np.asarray(learned_assets["latent_embeddings"], dtype=float),
        }
        for index, epoch_name in enumerate(
            ["Construction season", "EV adoption surge", "Service decentralization"]
        ):
            row, payload = self.evaluate_drift_epoch(
                epoch_name=epoch_name,
                original_bank=scenario_bank,
                original_fingerprints=original_fingerprints,
                master_baseline_topology=master_baseline_topology,
                seed=3100 + index,
                original_learned_assets=learned_assets,
            )
            drift_rows.append(row)
            latent_payloads[f"{epoch_name}_refresh"] = payload["refresh_embeddings"]
            latent_payloads[f"{epoch_name}_holdout"] = payload["holdout_embeddings"]

        drift_df = pd.DataFrame(drift_rows)
        table_3 = self.build_table_3(drift_df)

        # Save the main tables to the exact subsection folders.
        table_1_path = paths["s51_tab"] / "table_5_1_structural_fragility_summary.csv"
        bank_detail_path = paths["s51_tab"] / "table_5_1_accepted_scenario_detail.csv"
        table_2_path = paths["s52_tab"] / "table_5_2_controller_benchmark.csv"
        controller_detail_path = paths["s52_tab"] / "table_5_2_controller_detail.csv"
        table_3_path = paths["s53_tab"] / "table_5_3_drift_refresh_summary.csv"

        table_1.to_csv(table_1_path, index=False)
        bank_df.to_csv(bank_detail_path, index=False)
        table_2.to_csv(table_2_path, index=False)
        controller_df.to_csv(controller_detail_path, index=False)
        table_3.to_csv(table_3_path, index=False)

        # Save metadata and a lightweight manifest.
        metadata_path = paths["shared_data"] / "run_metadata.json"
        metadata = {
            "master_seed": self.master_seed,
            "families": self.families,
            "bank_size": len(scenario_bank),
            "history_size": len(generation_history),
            "dependency_versions": {
                "torch": getattr(torch, "__version__", "unknown"),
                "ripser": getattr(ripser, "__module__", "ripser"),
                "persim": getattr(wasserstein, "__module__", "persim"),
                "kmapper": getattr(km, "__version__", "unknown"),
                "toponetx": getattr(tnx, "__version__", "unknown"),
            },
            "learned_assets_summary": {
                "critical_edges": [list(edge) for edge in learned_assets["critical_edges"]],
                "critical_energy_nodes": learned_assets["critical_energy_nodes"],
                "critical_facilities": learned_assets["critical_facilities"],
            },
            "drift_epochs": drift_df.to_dict(orient="records"),
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        manifest_rows = []

        roadmap_path = paths["shared_fig"] / "figure_1_research_roadmap_autonomous_urban_simulator.png"
        self.plot_route_diagram(str(roadmap_path))
        manifest_rows.append(
            {"section": "shared", "type": "figure", "file": str(roadmap_path), "purpose": "Research-roadmap diagram for the whole chapter."}
        )

        figures_51 = self.plot_section_5_1(
            bank_df=bank_df,
            bank=scenario_bank,
            learned_assets=learned_assets,
            output_dir=str(paths["s51_fig"]),
        )
        figures_52 = self.plot_section_5_2(
            controller_df=controller_df,
            output_dir=str(paths["s52_fig"]),
        )
        figures_53 = self.plot_section_5_3(
            drift_df=drift_df,
            latent_payloads=latent_payloads,
            output_dir=str(paths["s53_fig"]),
        )

        for path in [table_1_path, bank_detail_path]:
            manifest_rows.append(
                {"section": "5.1", "type": "table", "file": str(path), "purpose": "Section 5.1 structural-fragility evidence."}
            )
        for path in figures_51.values():
            manifest_rows.append(
                {"section": "5.1", "type": "figure", "file": str(path), "purpose": "Section 5.1 structural-fragility figure."}
            )

        for path in [table_2_path, controller_detail_path]:
            manifest_rows.append(
                {"section": "5.2", "type": "table", "file": str(path), "purpose": "Section 5.2 controller benchmark evidence."}
            )
        for path in figures_52.values():
            manifest_rows.append(
                {"section": "5.2", "type": "figure", "file": str(path), "purpose": "Section 5.2 controller benchmark figure."}
            )

        manifest_rows.append(
            {"section": "5.3", "type": "table", "file": str(table_3_path), "purpose": "Section 5.3 drift-and-refresh evidence."}
        )
        for path in figures_53.values():
            manifest_rows.append(
                {"section": "5.3", "type": "figure", "file": str(path), "purpose": "Section 5.3 drift-and-refresh figure."}
            )

        manifest_path = paths["shared_data"] / "artifact_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        self.restore_original_state()

        artifact_dict = {
            "root": str(paths["root"]),
            "metadata": str(metadata_path),
            "manifest": str(manifest_path),
            "roadmap": str(roadmap_path),
            "table_5_1": str(table_1_path),
            "table_5_2": str(table_2_path),
            "table_5_3": str(table_3_path),
        }
        artifact_dict.update(figures_51)
        artifact_dict.update(figures_52)
        artifact_dict.update(figures_53)
        return artifact_dict


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    The defaults are chosen so that a reader can run the file directly and
    obtain a complete artifact tree without external configuration files.
    """
    parser = argparse.ArgumentParser(
        description="Generate section-aware chapter outputs for the autonomous urban scenario simulator."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "autonomous_urban_outputs_tda"),
        help="Root directory where the subsection-organized outputs will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Master seed for reproducible scenario generation and learning.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """
    args = parse_args()
    simulator = TopologicalAutonomousUrbanScenarioSimulator(master_seed=args.seed)
    artifacts = simulator.run_all(output_dir=args.output_root)

    print("Generated artifacts:")
    for label, path in artifacts.items():
        print(f" - {label}: {path}")


if __name__ == "__main__":
    main()
