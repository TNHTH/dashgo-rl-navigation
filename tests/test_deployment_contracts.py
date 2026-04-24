from __future__ import annotations

import numpy as np

from dashgo_rl.deployment.contracts import DashGoObservationContract, select_local_waypoint


def test_select_local_waypoint_matches_progressive_forward_rule() -> None:
    path_points = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert select_local_waypoint(path_points, lookahead_dist=0.6) == 4


def test_observation_contract_manifest_is_explicit() -> None:
    manifest = DashGoObservationContract().to_manifest()

    assert manifest["obs_dim"] == 246
    assert manifest["action_dim"] == 2
    assert manifest["action_semantics"] == "bounded_tanh_gaussian"
