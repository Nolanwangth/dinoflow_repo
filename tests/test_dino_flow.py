import torch

from lerobot.policies.dino_flow.configuration_dino_flow import DinoFlowConfig
from lerobot.policies.dino_flow.modeling_dino_flow import ActionDiT, DinoFlowPolicy


def _small_config() -> DinoFlowConfig:
    return DinoFlowConfig(
        horizon=5,
        n_action_steps=5,
        state_dim=4,
        action_dim=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        resampler_tokens=4,
        resampler_heads=4,
        image_resize_shapes={"camera_a": (32, 32), "camera_b": (32, 32)},
    )


def test_dino_flow_config_defaults_are_valid_without_dataset_features():
    config = _small_config()

    assert config.type == "dino_flow"
    assert config.drop_n_last_frames == 0
    assert config.action_delta_indices == [0, 1, 2, 3, 4]


def test_action_dit_forward_shape():
    config = _small_config()
    model = ActionDiT(config)
    noisy_action = torch.randn(2, config.horizon, config.action_dim)
    state = torch.randn(2, config.state_dim)
    visual_tokens = [torch.randn(2, 3, config.hidden_dim), torch.randn(2, 2, config.hidden_dim)]
    timestep = torch.rand(2)

    output = model(noisy_action, state, visual_tokens, timestep)

    assert output.shape == (2, config.horizon, config.action_dim)
    assert torch.isfinite(output).all()


def test_rtc_prefix_weights_fade_to_zero():
    weights = DinoFlowPolicy._rtc_prefix_weights(6, 1, 4, "cpu", torch.float32)

    torch.testing.assert_close(weights, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]))
