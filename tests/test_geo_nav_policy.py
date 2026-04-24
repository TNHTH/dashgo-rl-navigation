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


def test_action_mean_remains_latent_and_bounded_mean_is_exported() -> None:
    obs = torch.randn(4, 246)
    policy = GeoNavPolicy(
        obs=obs,
        obs_groups=None,
        num_actions=2,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
        init_noise_std=1.0,
    )

    policy.update_distribution(obs)
    latent_mean = policy.forward_actor(obs)

    assert torch.allclose(policy.action_mean, latent_mean)
    assert torch.allclose(policy.bounded_action_mean, torch.tanh(latent_mean))
    assert torch.all(policy.bounded_action_mean <= 1.0)
    assert torch.all(policy.bounded_action_mean >= -1.0)
