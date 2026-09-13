import numpy as np
import torch
from torch import nn

from deployment.client_mock import ACTION_HORIZON, CONTROL_DT, MockClient
from lerobot.datasets.sampler import FixedEpisodeSampler
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
    assert config.use_delta_action is False


def test_default_config_uses_absolute_action_and_shared_256_latent():
    config = DinoFlowConfig()

    assert config.use_delta_action is False
    assert config.hidden_dim == 256


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
    tokens = encoder.forward_tokens(state_history)
    assert output.shape == (2, 128)
    assert tokens.shape == (2, config.tactile_history_steps * 14, config.contact_token_dim)
    assert torch.isfinite(output).all()
    assert torch.isfinite(tokens).all()


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
    contact_tokens = torch.randn(
        2, config.tactile_history_steps * 14, config.contact_token_dim
    )

    output = model(
        noisy_action,
        state,
        visual_tokens,
        visual_valid_mask,
        timestep,
        contact_tokens=contact_tokens,
    )

    assert output.shape == (2, config.horizon, config.action_dim)
    assert torch.isfinite(output).all()


def test_contact_attention_is_zero_initialized():
    config = _small_config()
    model = ActionDiT(config).eval()
    assert model.blocks[-1].contact_cross_attn is not None
    noisy_action = torch.randn(2, config.horizon, config.action_dim)
    state = torch.randn(2, config.state_dim)
    visual_tokens = torch.randn(2, 5, config.hidden_dim)
    visual_valid_mask = torch.ones(2, 5, dtype=torch.bool)
    timestep = torch.rand(2)
    contact_tokens = torch.randn(
        2, config.tactile_history_steps * 14, config.contact_token_dim
    )

    without_tokens = model(noisy_action, state, visual_tokens, visual_valid_mask, timestep)
    with_tokens = model(
        noisy_action,
        state,
        visual_tokens,
        visual_valid_mask,
        timestep,
        contact_tokens=contact_tokens,
    )

    torch.testing.assert_close(with_tokens, without_tokens)

    with torch.no_grad():
        model.blocks[-1].contact_out.weight.fill_(0.01)
    after_learning = model(
        noisy_action,
        state,
        visual_tokens,
        visual_valid_mask,
        timestep,
        contact_tokens=contact_tokens,
    )
    assert not torch.equal(after_learning, without_tokens)


def test_camera_embedding_is_enabled_by_config():
    # Camera embeddings live on DinoFlowPolicy, not on the action-only module.
    config = _small_config()
    assert config.use_camera_embedding is True
    assert config.hidden_dim == 32


def test_rtc_prefix_weights_fade_to_zero():
    weights = DinoFlowPolicy._rtc_prefix_weights(6, 1, 4, "cpu", torch.float32)

    torch.testing.assert_close(weights, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]))


def test_rtc_prefix_weights_ignore_padded_previous_chunk_tail():
    weights = DinoFlowPolicy._rtc_prefix_weights(
        6, 0, 5, "cpu", torch.float32, available_length=2
    )

    torch.testing.assert_close(weights, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    delayed_weights = DinoFlowPolicy._rtc_prefix_weights(
        6, 3, 6, "cpu", torch.float32, available_length=2
    )
    torch.testing.assert_close(delayed_weights, torch.zeros(6))


def test_predict_action_chunk_preserves_explicit_zero_execution_horizon():
    policy = DinoFlowPolicy.__new__(DinoFlowPolicy)
    captured = {}

    def sample(batch, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 5, 4)

    policy._sample = sample
    policy.predict_action_chunk({}, execution_horizon=0)

    assert captured["execution_horizon"] == 0


def test_fixed_episode_sampler_is_deterministic_and_temporally_spread():
    sampler = FixedEpisodeSampler([0, 10, 25], [10, 25, 40], max_frames=9)

    indices = list(sampler)
    assert indices == [0, 4, 9, 10, 17, 24, 25, 32, 39]
    assert indices == list(FixedEpisodeSampler([0, 10, 25], [10, 25, 40], max_frames=9))


def test_deployment_history_grid_stays_at_training_rate():
    client = MockClient("127.0.0.1", 0, hz=10)
    client.state_samples.extend(
        (index / 10.0, np.full(646, index, dtype=np.float32)) for index in range(6)
    )

    _, target_times, source_times, source_reuse = client._resampled_state_history(0.5)

    np.testing.assert_allclose(np.diff(target_times), CONTROL_DT)
    assert all(source <= target for source, target in zip(source_times, target_times, strict=True))
    assert max(source_reuse) > 1


def test_deployment_drops_chunks_that_are_older_than_the_action_window():
    client = MockClient("127.0.0.1", 0, hz=30)
    old_chunk = np.full((30, 26), -1.0, dtype=np.float32)
    client.current_chunk = old_chunk.copy()
    client.current_idx = 20
    client.exec_counter = 3
    actions = np.zeros((ACTION_HORIZON, 26), dtype=np.float32)

    consumed, age, stale = client._install_action_chunk(
        actions, request_start_exec=0, observation_time=0.0, response_time=2.0
    )

    assert consumed == 3
    assert age == 2.0
    assert stale is True
    np.testing.assert_array_equal(client.current_chunk, old_chunk)
