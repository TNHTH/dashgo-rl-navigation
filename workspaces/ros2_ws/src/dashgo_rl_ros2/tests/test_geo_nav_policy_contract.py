import torch

from dashgo_rl.geo_nav_policy import GeoNavPolicy


@torch.no_grad()
def test_geo_nav_policy_forward_is_bounded():
    policy = GeoNavPolicy(obs=246, obs_groups=None, num_actions=2)
    obs = torch.randn(4, 246)
    actions = policy.forward(obs)

    assert actions.shape == (4, 2)
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)


@torch.no_grad()
def test_geo_nav_policy_log_prob_is_finite_for_bounded_actions():
    policy = GeoNavPolicy(obs=246, obs_groups=None, num_actions=2)
    obs = torch.randn(4, 246)
    actions = policy.act(obs)
    log_prob = policy.get_actions_log_prob(actions)

    assert actions.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert torch.isfinite(log_prob).all()
