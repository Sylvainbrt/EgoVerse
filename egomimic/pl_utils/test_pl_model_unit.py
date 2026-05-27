from functools import partial
from types import SimpleNamespace

import torch

from egomimic.pl_utils.pl_model import ModelWrapper


class _DummyAlgo:
    def __init__(self) -> None:
        self.nets = torch.nn.ModuleDict({"policy": torch.nn.Linear(1, 1)})
        self.domains = ["viperx_right_arm", "scale_bimanual"]


def _make_wrapper() -> ModelWrapper:
    wrapper = ModelWrapper(
        _DummyAlgo(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=None,
    )
    wrapper._trainer = SimpleNamespace(is_global_zero=False)
    return wrapper


def test_auto_exclude_filters_bad_scale_samples(monkeypatch) -> None:
    monkeypatch.setenv("EGOVERSE_AUTO_EXCLUDE_ACTION_MAX_ABS", "100")
    wrapper = _make_wrapper()

    raw_batch = {
        "viperx_right_arm": {
            "actions_joints": torch.ones(2, 5, 7),
        },
        "scale_bimanual": {
            "actions_cartesian": torch.tensor(
                [
                    [[[1.0]], [[2.0]]],
                    [[[5000.0]], [[4.0]]],
                    [[[3.0]], [[1.0]]],
                ]
            ),
            "metadata.episode_hash": ["good_a", "bad_a", "good_b"],
            "metadata.frame_idx": torch.tensor([11, 22, 33]),
            "aux_list": ["keep0", "drop1", "keep2"],
            "aux_tuple": ("keep0", "drop1", "keep2"),
        },
    }

    filtered = wrapper._auto_exclude_and_filter_raw_batch(raw_batch)

    assert filtered["scale_bimanual"]["actions_cartesian"].shape[0] == 2
    assert filtered["scale_bimanual"]["metadata.episode_hash"] == ["good_a", "good_b"]
    assert filtered["scale_bimanual"]["metadata.frame_idx"].tolist() == [11, 33]
    assert filtered["scale_bimanual"]["aux_list"] == ["keep0", "keep2"]
    assert filtered["scale_bimanual"]["aux_tuple"] == ("keep0", "keep2")
    assert wrapper.auto_exclusion_pending == {"scale_bimanual": {"bad_a"}}


def test_auto_exclude_drops_scale_domain_when_every_sample_is_bad(monkeypatch) -> None:
    monkeypatch.setenv("EGOVERSE_AUTO_EXCLUDE_ACTION_MAX_ABS", "100")
    wrapper = _make_wrapper()

    raw_batch = {
        "viperx_right_arm": {
            "actions_joints": torch.ones(2, 5, 7),
        },
        "scale_bimanual": {
            "actions_cartesian": torch.tensor(
                [
                    [[[5000.0]], [[4.0]]],
                    [[[7000.0]], [[8.0]]],
                ]
            ),
            "metadata.episode_hash": ["bad_a", "bad_b"],
        },
    }

    filtered = wrapper._auto_exclude_and_filter_raw_batch(raw_batch)

    assert "scale_bimanual" not in filtered
    assert wrapper.auto_exclusion_pending == {"scale_bimanual": {"bad_a", "bad_b"}}


def test_action_loss_is_rescaled_when_a_domain_is_dropped() -> None:
    wrapper = _make_wrapper()
    losses = {"action_loss": torch.tensor(5.0)}

    adjusted = wrapper._maybe_rescale_action_loss_for_active_domains(1, losses)

    assert float(adjusted["action_loss"]) == 10.0
