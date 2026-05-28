from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
            lambda filters, debug=False, **kwargs: [
                ("processed_v3/scale/hash_a", "hash_a"),
                ("s3://custom-bucket/processed_v3/scale/hash_b/", "hash_b"),
            ]
        ),
    )

    captured = {}

    def fake_build(self, episode_paths):
        captured["episode_paths"] = episode_paths
        return {
            episode_hash: str(episode_path)
            for episode_path, episode_hash in episode_paths
        }

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


def test_manifest_resolver_backfills_missing_entries_from_later_manifest_items(
    monkeypatch, tmp_path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
{
  "episodes": [
    {"processed_path": "s3://rldb/processed_v3/scale/missing_a.zarr/", "episode_hash": "missing_a"},
    {"processed_path": "s3://rldb/processed_v3/scale/present_b.zarr/", "episode_hash": "present_b"},
    {"processed_path": "s3://rldb/processed_v3/scale/present_c.zarr/", "episode_hash": "present_c"}
  ]
}
""".strip()
    )

    cache_dir = tmp_path / "cache"
    (cache_dir / "present_b").mkdir(parents=True)
    (cache_dir / "present_c").mkdir(parents=True)

    synced_batches = []

    def fake_sync(*, bucket_name, s3_paths, local_dir, numworkers=10):
        synced_batches.append([episode_hash for _, episode_hash in s3_paths])

    monkeypatch.setattr(
        zdm.S3EpisodeResolver,
        "_sync_s3_to_local",
        staticmethod(fake_sync),
    )
    monkeypatch.setattr(
        zdm.EpisodeResolver,
        "_build_zarr_datasets",
        lambda self, episode_paths: {
            episode_hash: str(episode_path)
            for episode_path, episode_hash in episode_paths
        },
    )

    resolver = zdm.ManifestEpisodeResolver(
        folder_path=cache_dir,
        manifest_path=manifest_path,
        key_map={},
        sync_missing=True,
        max_episodes=2,
    )

    datasets = resolver.resolve()

    assert list(datasets.keys()) == ["present_b", "present_c"]
    assert synced_batches == [["missing_a", "present_b"], ["present_c"]]


def test_manifest_resolver_local_only_keeps_first_requested_entries_strict(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
{
  "episodes": [
    {"processed_path": "s3://rldb/processed_v3/scale/missing_a.zarr/", "episode_hash": "missing_a"},
    {"processed_path": "s3://rldb/processed_v3/scale/present_b.zarr/", "episode_hash": "present_b"},
    {"processed_path": "s3://rldb/processed_v3/scale/present_c.zarr/", "episode_hash": "present_c"}
  ]
}
""".strip()
    )

    cache_dir = tmp_path / "cache"
    (cache_dir / "present_b").mkdir(parents=True)
    (cache_dir / "present_c").mkdir(parents=True)

    resolver = zdm.ManifestEpisodeResolver(
        folder_path=cache_dir,
        manifest_path=manifest_path,
        key_map={},
        sync_missing=False,
        max_episodes=2,
    )

    with pytest.raises(FileNotFoundError, match="missing_a"):
        resolver.resolve()


def test_s3_sync_skips_remote_episodes_marked_unavailable(
    monkeypatch, tmp_path
) -> None:
    zdm.S3EpisodeResolver._known_unavailable_remote_episodes = set()
    zdm.S3EpisodeResolver._mark_remote_episodes_unavailable(
        "rldb",
        [("s3://rldb/processed_v3/scale/missing_a.zarr/", "missing_a")],
    )

    monkeypatch.setattr(
        zdm.S3EpisodeResolver,
        "_episode_already_present",
        classmethod(lambda cls, local_dir, episode_hash: False),
    )
    monkeypatch.setattr(zdm, "load_env", lambda: None)
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")

    captured = {}

    def fake_run(cmd, check, env, capture_output, text):
        batch_path = Path(cmd[-1])
        captured["lines"] = batch_path.read_text().splitlines()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(zdm.subprocess, "run", fake_run)

    zdm.S3EpisodeResolver._sync_s3_to_local(
        bucket_name="rldb",
        s3_paths=[
            ("s3://rldb/processed_v3/scale/missing_a.zarr/", "missing_a"),
            ("s3://rldb/processed_v3/scale/present_b.zarr/", "present_b"),
        ],
        local_dir=tmp_path / "cache",
    )

    assert captured["lines"] == [
        f'sync "s3://rldb/processed_v3/scale/present_b.zarr/*" "{tmp_path / "cache" / "present_b"}/"'
    ]


def test_s3_sync_tolerates_no_object_found_partial_failures(
    monkeypatch, tmp_path
) -> None:
    zdm.S3EpisodeResolver._known_unavailable_remote_episodes = set()

    monkeypatch.setattr(
        zdm.S3EpisodeResolver,
        "_episode_already_present",
        classmethod(lambda cls, local_dir, episode_hash: False),
    )
    monkeypatch.setattr(zdm, "load_env", lambda: None)
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")

    def fake_run(cmd, check, env, capture_output, text):
        return SimpleNamespace(
            returncode=1,
            stdout='ERROR "sync s3://rldb/processed_v3/scale/missing_a.zarr/* /tmp/cache/missing_a/": no object found\n',
            stderr="",
        )

    monkeypatch.setattr(zdm.subprocess, "run", fake_run)

    zdm.S3EpisodeResolver._sync_s3_to_local(
        bucket_name="rldb",
        s3_paths=[("s3://rldb/processed_v3/scale/missing_a.zarr/", "missing_a")],
        local_dir=tmp_path / "cache",
    )


def test_manifest_resolver_proceeds_with_available_subset_when_manifest_exhausted(
    monkeypatch, tmp_path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
{
  "episodes": [
    {"processed_path": "s3://rldb/processed_v3/scale/present_a.zarr/", "episode_hash": "present_a"},
    {"processed_path": "s3://rldb/processed_v3/scale/missing_b.zarr/", "episode_hash": "missing_b"}
  ]
}
""".strip()
    )

    cache_dir = tmp_path / "cache"
    (cache_dir / "present_a").mkdir(parents=True)

    monkeypatch.setattr(
        zdm.S3EpisodeResolver,
        "_sync_s3_to_local",
        staticmethod(lambda **kwargs: None),
    )
    monkeypatch.setattr(
        zdm.EpisodeResolver,
        "_build_zarr_datasets",
        lambda self, episode_paths: {
            episode_hash: str(episode_path)
            for episode_path, episode_hash in episode_paths
        },
    )

    resolver = zdm.ManifestEpisodeResolver(
        folder_path=cache_dir,
        manifest_path=manifest_path,
        key_map={},
        sync_missing=True,
        max_episodes=2,
    )

    datasets = resolver.resolve()

    assert list(datasets.keys()) == ["present_a"]
