from __future__ import annotations

import torch

from dashgo_rl.geo_nav_policy import GeoNavPolicy


def test_act_inference_accepts_dict_observations() -> None:
    obs = {"policy": torch.randn(4, 246)}
    policy = GeoNavPolicy(
        obs=obs,
        obs_groups=None,
        num_actions=2,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
        init_noise_std=1.0,
    )

    actions = policy.act_inference(obs)

    assert actions.shape == (4, 2)
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)
