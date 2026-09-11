import torch
from torch import nn

from lerobot.policies.dino_flow.configuration_dino_flow import DinoFlowConfig
from lerobot.policies.dino_flow.modeling_dino_flow import (
    ActionDiT,
    ContactHistoryEncoder,
    DinoFlowPolicy,
    DinoVisionEncoder,
)


def _small_config() -> DinoFlowConfig:
    return DinoFlowConfig(
        horizon=5,
        n_action_steps=5,
        state_dim=4,
        action_dim=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        image_resize_shapes={"camera_a": (32, 32), "camera_b": (32, 32)},
    )


def test_dino_flow_config_defaults_are_valid_without_dataset_features():
    config = _small_config()

    assert config.type == "dino_flow"
    assert config.drop_n_last_frames == 0
    assert config.action_delta_indices == [0, 1, 2, 3, 4]


def test_default_camera_targets_preserve_wrist_width():
    config = DinoFlowConfig()

    assert config.image_resize_shapes["observation.images.base_0_rgb"] == (480, 768)
    assert config.image_resize_shapes["observation.images.left_wrist_0_rgb"] == (480, 832)
    assert config.image_resize_shapes["observation.images.right_wrist_0_rgb"] == (480, 832)


def test_contact_history_layout_and_output():
    config = DinoFlowConfig()
    assert config.contact_history_delta_indices == [-5, -4, -3, -2, -1, 0]
    encoder = ContactHistoryEncoder(config)
    state_history = torch.randn(2, config.tactile_history_steps, config.observation_state_dim)
    output = encoder(state_history)
    assert output.shape == (2, 128)
    assert torch.isfinite(output).all()


def test_contact_history_is_causal_and_trainable():
    config = DinoFlowConfig()
    encoder = ContactHistoryEncoder(config)
    state_history = torch.zeros(1, config.tactile_history_steps, config.observation_state_dim)
    state_history[:, -1, config.tactile_state_offset] = 1.0
    first = encoder(state_history)
    state_history[:, 0, config.tactile_state_offset] = 100.0
    second = encoder(state_history)
    assert not torch.equal(first, second)
    second.sum().backward()
    assert any(param.grad is not None for param in encoder.parameters())


def test_wrist_preprocess_center_crops_only_eight_pixels_each_side():
    class FakeDino(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"patch_size": 16})()

    encoder = DinoVisionEncoder.__new__(DinoVisionEncoder)
    nn.Module.__init__(encoder)
    encoder.model = FakeDino()
    encoder.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    encoder.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    source = torch.linspace(0.0, 1.0, 848).view(1, 1, 1, 848).expand(1, 3, 480, 848)
    processed, valid_mask = encoder._preprocess(source, (480, 832))
    expected = (source[..., 8:840] - encoder.mean) / encoder.std

    assert processed.shape == (1, 3, 480, 832)
    torch.testing.assert_close(processed, expected)
    assert valid_mask.shape == (1, 30 * 52)
    assert valid_mask.all()


def test_action_dit_forward_shape():
    config = _small_config()
    model = ActionDiT(config)
    noisy_action = torch.randn(2, config.horizon, config.action_dim)
    state = torch.randn(2, config.state_dim)
    visual_tokens = torch.randn(2, 5, config.hidden_dim)
    visual_valid_mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)
    timestep = torch.rand(2)

    output = model(noisy_action, state, visual_tokens, visual_valid_mask, timestep)

    assert output.shape == (2, config.horizon, config.action_dim)
    assert torch.isfinite(output).all()


def test_action_dit_has_no_camera_embedding():
    model = ActionDiT(_small_config())

    assert not hasattr(model, "camera_embeddings")


def test_rtc_prefix_weights_fade_to_zero():
    weights = DinoFlowPolicy._rtc_prefix_weights(6, 1, 4, "cpu", torch.float32)

    torch.testing.assert_close(weights, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]))
