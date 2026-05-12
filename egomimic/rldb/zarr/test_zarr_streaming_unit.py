from pathlib import Path

import numpy as np

from egomimic.rldb.zarr import zarr_dataset_multi as zdm


def test_s3_streaming_resolver_builds_remote_episode_paths(monkeypatch) -> None:
    resolver = zdm.S3StreamingEpisodeResolver(
        folder_path=Path("/tmp/unused"),
        key_map={},
    )

    monkeypatch.setattr(
        zdm.S3StreamingEpisodeResolver,
        "_get_filtered_paths",
        staticmethod(
            lambda filters, debug=False: [
                ("processed_v3/scale/hash_a", "hash_a"),
                ("s3://custom-bucket/processed_v3/scale/hash_b/", "hash_b"),
            ]
        ),
    )

    captured = {}

    def fake_build(self, episode_paths):
        captured["episode_paths"] = episode_paths
        return {episode_hash: str(episode_path) for episode_path, episode_hash in episode_paths}

    monkeypatch.setattr(zdm.EpisodeResolver, "_build_zarr_datasets", fake_build)

    datasets = resolver.resolve(filters={"episode_hash": "hash_a"})

    assert datasets == {
        "hash_a": "s3://rldb/processed_v3/scale/hash_a",
        "hash_b": "s3://custom-bucket/processed_v3/scale/hash_b",
    }
    assert captured["episode_paths"] == [
        ("s3://rldb/processed_v3/scale/hash_a", "hash_a"),
        ("s3://custom-bucket/processed_v3/scale/hash_b", "hash_b"),
    ]


def test_zarr_episode_reopens_remote_store_after_pickle_roundtrip(monkeypatch) -> None:
    open_calls = []

    class _FakeArray:
        def __getitem__(self, item):
            return np.array([10, 11, 12])[item]

    class _FakeGroup:
        attrs = {"features": {"arr": {"dtype": "int32"}}, "total_frames": 3}

        def __getitem__(self, key):
            assert key == "arr"
            return _FakeArray()

    def fake_open_group(*args, **kwargs):
        open_calls.append((args, kwargs))
        return _FakeGroup()

    monkeypatch.setattr(zdm.zarr, "open_group", fake_open_group)
    monkeypatch.setattr(
        zdm.ZarrEpisode,
        "_build_remote_store",
        staticmethod(lambda path: f"store-for:{path}"),
    )

    episode = zdm.ZarrEpisode("s3://bucket/episode")
    assert len(open_calls) == 1

    state = episode.__getstate__()
    restored = object.__new__(zdm.ZarrEpisode)
    restored.__setstate__(state)

    result = restored.read({"arr": (1, None)})

    assert result == {"arr": 11}
    assert len(open_calls) == 2
    assert open_calls[0][1] == {"store": "store-for:s3://bucket/episode", "mode": "r"}
    assert open_calls[1][1] == {"store": "store-for:s3://bucket/episode", "mode": "r"}
