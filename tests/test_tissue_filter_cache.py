"""Tests for the tissue-filter JSON cache.

Parallel in spirit to ``test_flat_field_cache.py``: save/load roundtrip, the
full invalidation matrix on every metadata key, corruption resilience, and
the ``--flat-field-cache-dir`` co-location behavior. The tissue cache is
simpler (plain JSON, no advisory lock) so the surface is smaller.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from xldvp_seg.preprocessing import tissue_filter_cache as tc


@pytest.fixture
def sample_meta(tmp_path):
    czi = tmp_path / "slide.czi"
    czi.write_bytes(b"x" * 1024)
    args = SimpleNamespace(czi_path=str(czi), tile_size=3000, tile_overlap=0.25)
    return tc.build_cache_meta(
        args,
        tissue_channel=2,
        modality="fluorescence",
        manual_threshold=None,
        n_all_tiles=5700,
    )


@pytest.fixture
def sample_tiles():
    return [{"x": x, "y": y} for x in range(0, 9000, 3000) for y in range(0, 6000, 3000)]


# ---------------------------------------------------------------------------
# Meta builder
# ---------------------------------------------------------------------------


class TestBuildCacheMeta:
    def test_captures_czi_identity(self, tmp_path):
        czi = tmp_path / "slide.czi"
        czi.write_bytes(b"abc" * 100)
        args = SimpleNamespace(czi_path=str(czi), tile_size=3000, tile_overlap=0.1)
        meta = tc.build_cache_meta(
            args,
            tissue_channel=2,
            modality="fluorescence",
            manual_threshold=None,
            n_all_tiles=100,
        )
        assert meta["czi_path"] == str(czi)
        assert meta["czi_size"] == 300
        assert meta["czi_mtime"] > 0.0
        assert meta["tile_size"] == 3000
        assert meta["tile_overlap"] == 0.1
        assert meta["tissue_channel"] == 2
        assert meta["modality"] == "fluorescence"
        assert meta["manual_threshold"] is None
        assert meta["n_all_tiles"] == 100
        assert meta["algorithm_version"] == tc.ALGORITHM_VERSION

    def test_missing_czi_yields_zero_stat_not_crash(self):
        args = SimpleNamespace(czi_path="/does/not/exist.czi", tile_size=3000, tile_overlap=0.1)
        meta = tc.build_cache_meta(
            args,
            tissue_channel=0,
            modality="brightfield",
            manual_threshold=15.0,
            n_all_tiles=50,
        )
        assert meta["czi_mtime"] == 0.0
        assert meta["czi_size"] == 0
        assert meta["manual_threshold"] == 15.0


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_threshold_and_tiles(self, tmp_path, sample_meta, sample_tiles):
        path = tmp_path / tc.CACHE_FILENAME
        tc.save(path, variance_threshold=42.5, tissue_tiles=sample_tiles, metadata=sample_meta)
        assert path.exists()

        loaded = tc.load(path, sample_meta)
        assert loaded is not None
        threshold, tiles = loaded
        assert threshold == pytest.approx(42.5)
        assert len(tiles) == len(sample_tiles)
        for got, want in zip(tiles, sample_tiles, strict=True):
            assert got["x"] == want["x"]
            assert got["y"] == want["y"]

    def test_save_is_atomic_no_partial(self, tmp_path, sample_meta, sample_tiles):
        path = tmp_path / tc.CACHE_FILENAME
        tc.save(path, 42.5, sample_tiles, sample_meta)
        files = sorted(f.name for f in tmp_path.iterdir() if tc.CACHE_FILENAME in f.name)
        # Only the final file; no .tmp, no .partial
        assert files == [tc.CACHE_FILENAME], f"unexpected files: {files}"

    def test_save_extracts_only_xy_keys(self, tmp_path, sample_meta):
        """Tiles may carry extra keys; the cache only persists x and y."""
        path = tmp_path / tc.CACHE_FILENAME
        rich_tiles = [{"x": 0, "y": 0, "foo": "bar", "w": 3000, "h": 3000}]
        tc.save(path, 10.0, rich_tiles, sample_meta)
        loaded = tc.load(path, sample_meta)
        assert loaded is not None
        _, tiles = loaded
        # Round-trip tiles have only x/y (not foo/w/h)
        assert set(tiles[0].keys()) == {"x", "y"}


# ---------------------------------------------------------------------------
# Cache validator
# ---------------------------------------------------------------------------


class TestCacheValidator:
    def _save(self, tmp_path, meta, threshold=42.5, tiles=None):
        path = tmp_path / tc.CACHE_FILENAME
        tc.save(path, threshold, tiles or [{"x": 0, "y": 0}], meta)
        return path

    def test_hit_on_matching_metadata(self, tmp_path, sample_meta):
        path = self._save(tmp_path, sample_meta)
        assert tc.load(path, sample_meta) is not None

    def test_miss_on_missing_file(self, tmp_path, sample_meta):
        path = tmp_path / tc.CACHE_FILENAME
        assert tc.load(path, sample_meta) is None

    def test_miss_on_none_cache_path(self, sample_meta):
        assert tc.load(None, sample_meta) is None

    @pytest.mark.parametrize(
        "key,new_value",
        [
            ("czi_path", "/different/slide.czi"),
            ("czi_size", 99999999),
            ("tile_size", 2048),
            ("tile_overlap", 0.1),
            ("tissue_channel", 4),
            ("modality", "brightfield"),
            ("manual_threshold", 20.0),
            ("n_all_tiles", 9999),
            ("algorithm_version", "99.99"),
        ],
    )
    def test_miss_on_any_meta_key_change(self, tmp_path, sample_meta, key, new_value):
        path = self._save(tmp_path, sample_meta)
        mutated = {**sample_meta, key: new_value}
        assert tc.load(path, mutated) is None, f"should invalidate on {key} change"

    def test_tolerates_sub_second_mtime_drift(self, tmp_path, sample_meta):
        path = self._save(tmp_path, sample_meta)
        drifted = {**sample_meta, "czi_mtime": sample_meta["czi_mtime"] + 0.4}
        assert tc.load(path, drifted) is not None

    def test_miss_on_large_mtime_drift(self, tmp_path, sample_meta):
        path = self._save(tmp_path, sample_meta)
        rewritten = {**sample_meta, "czi_mtime": sample_meta["czi_mtime"] + 3600.0}
        assert tc.load(path, rewritten) is None


# ---------------------------------------------------------------------------
# Corruption resilience
# ---------------------------------------------------------------------------


class TestCorruptionResilience:
    def test_truncated_json_treated_as_miss(self, tmp_path, sample_meta):
        path = tmp_path / tc.CACHE_FILENAME
        path.write_text('{"variance_threshold": 42.5, "tissue_tiles":')  # truncated
        assert tc.load(path, sample_meta) is None

    def test_random_garbage_treated_as_miss(self, tmp_path, sample_meta):
        path = tmp_path / tc.CACHE_FILENAME
        path.write_bytes(b"\x00\xff" * 50)
        assert tc.load(path, sample_meta) is None

    def test_non_object_json_treated_as_miss(self, tmp_path, sample_meta):
        path = tmp_path / tc.CACHE_FILENAME
        path.write_text("[1, 2, 3]")  # valid JSON, wrong shape
        assert tc.load(path, sample_meta) is None

    def test_missing_payload_fields_treated_as_miss(self, tmp_path, sample_meta):
        path = tmp_path / tc.CACHE_FILENAME
        # Valid meta but missing variance_threshold + tissue_tiles
        path.write_text(json.dumps({"__meta__": sample_meta}))
        assert tc.load(path, sample_meta) is None


# ---------------------------------------------------------------------------
# File contents sanity (for downstream tooling)
# ---------------------------------------------------------------------------


def test_cache_file_is_valid_json(tmp_path, sample_meta, sample_tiles):
    path = tmp_path / tc.CACHE_FILENAME
    tc.save(path, 12.3, sample_tiles, sample_meta)
    # Should be plain JSON (not a binary format), parseable by the stdlib.
    with path.open() as f:
        parsed = json.load(f)
    assert parsed["variance_threshold"] == 12.3
    assert len(parsed["tissue_tiles"]) == len(sample_tiles)
    assert parsed["__meta__"]["algorithm_version"] == tc.ALGORITHM_VERSION


def test_cache_filename_constant_stable():
    """Downstream tools (viewers, dashboards) may grep for this filename."""
    assert tc.CACHE_FILENAME == "tissue_filter.json"
