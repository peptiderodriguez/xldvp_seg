"""Tests for xldvp_seg.lmd.ms_queue — Thermo MS queue CSV builder."""

import json
import logging

import pandas as pd
import pytest

from xldvp_seg.exceptions import ConfigError
from xldvp_seg.lmd.ms_queue import (
    ThermoQueueConfig,
    build_thermo_queues,
    detect_quadrant,
    map_384_to_96,
    to_windows_path,
)
from xldvp_seg.lmd.well_plate import generate_quadrant_serpentine

# ---------------------------------------------------------------------------
# detect_quadrant
# ---------------------------------------------------------------------------


class TestDetectQuadrant:
    @pytest.mark.parametrize(
        "well,expected",
        [
            ("B2", "B2"),  # even row + even col
            ("N22", "B2"),
            ("B3", "B3"),  # even row + odd col
            ("B23", "B3"),
            ("C2", "C2"),  # odd row + even col
            ("O22", "C2"),
            ("C3", "C3"),  # odd row + odd col
            ("O23", "C3"),
            ("b4", "B2"),  # case-insensitive
        ],
    )
    def test_known_wells(self, well, expected):
        assert detect_quadrant(well) == expected

    @pytest.mark.parametrize("well", ["A1", "A24", "P1", "P24"])
    def test_outer_row_rejected(self, well):
        with pytest.raises(ConfigError, match="outer"):
            detect_quadrant(well)

    @pytest.mark.parametrize("well", ["B1", "B24", "O1", "O24"])
    def test_outer_col_rejected(self, well):
        with pytest.raises(ConfigError, match="outer"):
            detect_quadrant(well)

    @pytest.mark.parametrize("well", ["", "B", "Z9", "22B", "BB2", None])
    def test_malformed(self, well):
        with pytest.raises(ConfigError):
            detect_quadrant(well)


# ---------------------------------------------------------------------------
# map_384_to_96
# ---------------------------------------------------------------------------


class TestMap384To96:
    @pytest.mark.parametrize(
        "well,expected_96",
        [
            ("B2", "A1"),
            ("B4", "A2"),
            ("N22", "G11"),
            ("B3", "A1"),  # B3 quadrant: col 3 -> 96 col 1
            ("B23", "A11"),
            ("N23", "G11"),
            ("C2", "A1"),
            ("O22", "G11"),
            ("C3", "A1"),
            ("O23", "G11"),
        ],
    )
    def test_known_mappings(self, well, expected_96):
        assert map_384_to_96(well) == expected_96

    @pytest.mark.parametrize("quadrant", ["B2", "B3", "C2", "C3"])
    def test_quadrant_covers_all_96_positions(self, quadrant):
        """Every 77-well quadrant must map to exactly A1..G11 (no collisions)."""
        wells = generate_quadrant_serpentine(quadrant)
        mapped = [map_384_to_96(w) for w in wells]
        expected = {f"{r}{c}" for r in "ABCDEFG" for c in range(1, 12)}
        assert set(mapped) == expected
        assert len(mapped) == len(set(mapped))  # no duplicates


# ---------------------------------------------------------------------------
# to_windows_path
# ---------------------------------------------------------------------------


class TestToWindowsPath:
    def test_none(self):
        assert to_windows_path(None) == ""

    def test_empty(self):
        assert to_windows_path("") == ""

    def test_forward_to_back(self):
        assert to_windows_path("C:/Xcalibur/foo") == "C:\\Xcalibur\\foo"

    def test_already_windows(self):
        win = "C:\\Xcalibur\\methods\\foo"
        assert to_windows_path(win) == win

    def test_unix_prefix_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = to_windows_path("/Volumes/pool-mann/methods/foo")
        assert result == "\\Volumes\\pool-mann\\methods\\foo"
        assert any("Unix path" in r.message for r in caplog.records)

    def test_unix_prefix_warn_off(self, caplog):
        with caplog.at_level(logging.WARNING):
            to_windows_path("/Users/foo/bar", warn_unix=False)
        assert not any("Unix path" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# build_thermo_queues
# ---------------------------------------------------------------------------


def _sample_df(plates=(1,), wells_per_plate=None):
    """Tiny synthetic replicate manifest with B2 + B3 samples."""
    wells_per_plate = wells_per_plate or {
        "B2": ["B2", "B4", "D2", "F6"],
        "B3": ["B3", "D5", "F7"],
    }
    rows = []
    for plate in plates:
        for quad, wells in wells_per_plate.items():
            for i, w in enumerate(wells):
                rows.append(
                    {
                        "plate": plate,
                        "well": w,
                        "slide": f"slide{plate}{quad}{i}",
                        "bone": "femur" if i % 2 == 0 else "humerus",
                        "replicate": i + 1,
                    }
                )
    return pd.DataFrame(rows)


def _base_config(**overrides):
    defaults = dict(
        file_name_template="{date}_PROJ_{slide}_{bone}_rep{replicate}_{well_384}_{well_96}",
        autosampler_slots={"B2": 2, "B3": 3},
        date="20260423",
        shuffle=True,
        shuffle_seed=42,
    )
    defaults.update(overrides)
    return ThermoQueueConfig(**defaults)


class TestBuildThermoQueues:
    def test_single_plate_writes_per_box_and_key(self, tmp_path):
        df = _sample_df(plates=(1,))
        outputs = build_thermo_queues(df, _base_config(), tmp_path)

        assert "B2" in outputs and "B3" in outputs
        assert "_key_csv" in outputs and "_key_json" in outputs

        b2 = outputs["B2"].read_text().splitlines()
        assert b2[0] == "Bracket Type=4"
        assert b2[1] == "File Name,Path,Instrument Method,Position"
        # 4 B2 samples in the fixture
        assert len(b2) == 2 + 4

        b3 = outputs["B3"].read_text().splitlines()
        assert len(b3) == 2 + 3

    def test_positions_match_slot_and_well_96(self, tmp_path):
        df = _sample_df(plates=(1,))
        outputs = build_thermo_queues(df, _base_config(shuffle=False), tmp_path)

        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        # B2 well "B2" -> well_96 A1 -> position S2:A1
        positions = set(b2["Position"])
        assert all(p.startswith("S2:") for p in positions)
        assert "S2:A1" in positions  # B2->A1
        assert "S2:A2" in positions  # B4->A2

        b3 = pd.read_csv(outputs["B3"], skiprows=1)
        assert all(p.startswith("S3:") for p in b3["Position"])

    def test_shuffle_deterministic_same_seed(self, tmp_path):
        df = _sample_df(plates=(1,))
        out1 = build_thermo_queues(df, _base_config(shuffle_seed=7), tmp_path / "a")
        out2 = build_thermo_queues(df, _base_config(shuffle_seed=7), tmp_path / "b")
        assert out1["B2"].read_text() == out2["B2"].read_text()

    def test_shuffle_differs_across_boxes(self, tmp_path):
        # Build a larger fixture so shuffle actually reorders.
        wells = {
            "B2": [f"B{c}" for c in range(2, 23, 2)],
            "B3": [f"B{c}" for c in range(3, 24, 2)],
        }
        df = _sample_df(wells_per_plate=wells)
        outputs = build_thermo_queues(df, _base_config(), tmp_path)
        b2_first = pd.read_csv(outputs["B2"], skiprows=1)["File Name"].iloc[0]
        b3_first = pd.read_csv(outputs["B3"], skiprows=1)["File Name"].iloc[0]
        # Independent shuffle streams -> different first rows (with near certainty).
        assert b2_first != b3_first

    def test_multi_plate_writes_four_boxes(self, tmp_path):
        df = _sample_df(plates=(1, 2))
        outputs = build_thermo_queues(df, _base_config(), tmp_path)
        assert set(outputs) >= {"plate1_B2", "plate1_B3", "plate2_B2", "plate2_B3"}

    def test_empty_marker_kept_with_custom_template(self, tmp_path):
        df = _sample_df(plates=(1,))
        df.loc[df["well"] == "B2", "slide"] = "EMPTY"
        cfg = _base_config(
            empty_marker=("slide", "EMPTY"),
            empty_file_name_template="{date}_BLANK_{quadrant}_{well_96}",
        )
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        blanks = b2[b2["File Name"].str.contains("BLANK")]
        assert len(blanks) == 1
        assert "20260423_BLANK_B2_A1" in set(blanks["File Name"])

    def test_missing_slot_for_present_quadrant_raises(self, tmp_path):
        df = _sample_df(plates=(1,))
        cfg = _base_config(autosampler_slots={"B2": 2})  # B3 absent but B3 samples present
        with pytest.raises(ConfigError, match="B3"):
            build_thermo_queues(df, cfg, tmp_path)

    def test_combined_writes_both(self, tmp_path):
        df = _sample_df(plates=(1,))
        outputs = build_thermo_queues(df, _base_config(), tmp_path, combined=True)
        assert "_combined" in outputs
        # per-box files still exist
        assert outputs["B2"].exists() and outputs["B3"].exists()
        combined = pd.read_csv(outputs["_combined"], skiprows=1)
        assert len(combined) == 4 + 3  # sum of B2 + B3

    def test_duplicate_filename_raises(self, tmp_path):
        # Template that collides: only {slide} used, and two rows share slide.
        df = _sample_df(plates=(1,))
        df["slide"] = "SAME"
        cfg = _base_config(file_name_template="{slide}")
        with pytest.raises(ConfigError, match="duplicate"):
            build_thermo_queues(df, cfg, tmp_path)

    def test_collision_no_partial_files(self, tmp_path):
        """Phase 1.6: ConfigError on duplicate File Name must leave output empty."""
        df = _sample_df(plates=(1,))
        df["slide"] = "SAME"
        cfg = _base_config(file_name_template="{slide}")
        with pytest.raises(ConfigError, match="duplicate"):
            build_thermo_queues(df, cfg, tmp_path)
        csv_files = list(tmp_path.glob("*.csv"))
        assert csv_files == [], f"Partial files written before validation: {csv_files}"

    def test_empty_dataframe_raises(self, tmp_path):
        """Phase 4b: zero-row input must raise ConfigError with clear message."""
        df = pd.DataFrame(columns=["plate", "well", "slide", "bone", "replicate"])
        cfg = _base_config()
        with pytest.raises(ConfigError, match="no rows"):
            build_thermo_queues(df, cfg, tmp_path)

    def test_all_blank_input(self, tmp_path):
        """Phase 4b: every row matches empty_marker → only BLANK rows output."""
        df = _sample_df(plates=(1,))
        df["slide"] = "EMPTY"
        cfg = _base_config(
            empty_marker=("slide", "EMPTY"),
            empty_file_name_template="{date}_BLANK_{quadrant}_{well_96}",
        )
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        assert all(b2["File Name"].str.contains("BLANK"))
        key = pd.read_csv(outputs["_key_csv"])
        assert key["is_empty"].all()

    def test_single_row_input(self, tmp_path):
        """Phase 4b: single-row DataFrame must produce valid output."""
        df = pd.DataFrame(
            [{"plate": 1, "well": "B2", "slide": "s1", "bone": "femur", "replicate": 1}]
        )
        cfg = _base_config()
        outputs = build_thermo_queues(df, cfg, tmp_path)
        assert "B2" in outputs
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        assert len(b2) == 1

    def test_pool_exhaustion_separators_plus_interspersed(self, tmp_path):
        """Phase 4b: n_separators + n_interspersed > pool must raise up front."""
        rows = []
        for well, shapes in [
            ("B2", 1),
            ("B4", 1),
            ("B6", 10),
            ("B8", 10),
            ("B10", 100),
            ("B12", 100),
        ]:
            rows.append(
                {
                    "plate": 1,
                    "well": well,
                    "slide": f"s{well}",
                    "bone": "x",
                    "replicate": 1,
                    "shapes": shapes,
                }
            )
        df = pd.DataFrame(rows)
        cfg = _base_config(
            group_by_column="shapes",
            group_separator_blanks=True,
            interspersed_blanks=12,
        )
        with pytest.raises(ConfigError):
            build_thermo_queues(df, cfg, tmp_path)
        assert list(tmp_path.glob("*.csv")) == []

    def test_unknown_template_field_raises(self, tmp_path):
        df = _sample_df(plates=(1,))
        cfg = _base_config(file_name_template="{date}_{nosuchfield}_{well_384}")
        with pytest.raises(ConfigError, match="nosuchfield"):
            build_thermo_queues(df, cfg, tmp_path)

    def test_ms_method_none_warns_and_empty(self, tmp_path, caplog):
        df = _sample_df(plates=(1,))
        with caplog.at_level(logging.WARNING):
            outputs = build_thermo_queues(df, _base_config(ms_method=None), tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        assert (b2["Instrument Method"].fillna("") == "").all()
        assert any("ms_method is None" in r.message for r in caplog.records)

    def test_ms_method_string_windows_formatted(self, tmp_path):
        df = _sample_df(plates=(1,))
        cfg = _base_config(ms_method="C:/Xcalibur/methods/foo")
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        assert (b2["Instrument Method"] == "C:\\Xcalibur\\methods\\foo").all()

    def test_ms_method_callable_per_row(self, tmp_path):
        df = _sample_df(plates=(1,))
        cfg = _base_config(
            ms_method=lambda row: f"C:/methods/{row['bone']}",
        )
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        methods = set(b2["Instrument Method"])
        assert methods == {"C:\\methods\\femur", "C:\\methods\\humerus"}

    def test_sample_key_contains_all_metadata(self, tmp_path):
        df = _sample_df(plates=(1,))
        outputs = build_thermo_queues(df, _base_config(), tmp_path)
        key = pd.read_csv(outputs["_key_csv"])
        # every input column preserved
        for col in ("plate", "well", "slide", "bone", "replicate"):
            assert col in key.columns
        # derived columns present
        for col in (
            "File Name",
            "well_384",
            "well_96",
            "quadrant",
            "slot",
            "Position",
            "box_key",
            "box_row_index",
            "is_empty",
            "instrument_method",
            "queue_date",
        ):
            assert col in key.columns
        assert len(key) == len(df)

        key_json = json.loads(outputs["_key_json"].read_text())
        assert len(key_json) == len(df)
        assert set(key_json[0]) == set(key.columns)

    def test_sample_key_column_order_file_name_first(self, tmp_path):
        df = _sample_df(plates=(1,))
        outputs = build_thermo_queues(df, _base_config(), tmp_path)
        key = pd.read_csv(outputs["_key_csv"])
        assert list(key.columns)[0] == "File Name"

    def test_key_row_order_matches_combined(self, tmp_path):
        df = _sample_df(plates=(1, 2))
        outputs = build_thermo_queues(df, _base_config(), tmp_path, combined=True)
        combined = pd.read_csv(outputs["_combined"], skiprows=1)
        key = pd.read_csv(outputs["_key_csv"])
        # File Name order must match, row for row.
        assert list(key["File Name"]) == list(combined["File Name"])

    def test_multi_plate_with_empties_and_combined(self, tmp_path):
        df = _sample_df(plates=(1, 2))
        df.loc[df["well"] == "B2", "slide"] = "EMPTY"
        cfg = _base_config(empty_marker=("slide", "EMPTY"))
        outputs = build_thermo_queues(df, cfg, tmp_path, combined=True)
        # Each plate has its own B2 blank.
        assert "plate1_B2" in outputs and "plate2_B2" in outputs
        combined = pd.read_csv(outputs["_combined"], skiprows=1)
        blanks = combined[combined["File Name"].str.contains("BLANK")]
        assert len(blanks) == 2  # one per plate

    def test_bracketing_blanks_per_box_counts(self, tmp_path):
        """Each per-box CSV gains n lead + n trail BLANK rows."""
        df = _sample_df(plates=(1,))
        cfg = _base_config(bracketing_blanks=3)
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        # 4 real B2 samples + 3 lead + 3 trail = 10
        assert len(b2) == 4 + 6
        # lead positions = S2:H1..H3, trail = S2:H10..H12
        assert b2["Position"].iloc[0] == "S2:H1"
        assert b2["Position"].iloc[1] == "S2:H2"
        assert b2["Position"].iloc[2] == "S2:H3"
        assert b2["Position"].iloc[-3] == "S2:H10"
        assert b2["Position"].iloc[-1] == "S2:H12"

    def test_bracketing_blanks_combined_single_junction(self, tmp_path):
        """Combined drops intermediate trailing; only one set of n between boxes."""
        df = _sample_df(plates=(1,))
        cfg = _base_config(bracketing_blanks=3)
        outputs = build_thermo_queues(df, cfg, tmp_path, combined=True)
        combined = pd.read_csv(outputs["_combined"], skiprows=1)
        # 4 B2 real + 3 B2 lead + 3 B3 lead (single junction set) + 3 B3 real + 3 B3 trail = 16
        assert len(combined) == 3 + 4 + 3 + 3 + 3
        # first 3 rows = B2 lead
        assert list(combined["Position"].iloc[:3]) == ["S2:H1", "S2:H2", "S2:H3"]
        # last 3 rows = B3 trail
        assert list(combined["Position"].iloc[-3:]) == ["S3:H10", "S3:H11", "S3:H12"]

    def test_bracketing_blanks_validation(self, tmp_path):
        df = _sample_df(plates=(1,))
        with pytest.raises(ConfigError, match="bracketing_blanks"):
            build_thermo_queues(df, _base_config(bracketing_blanks=7), tmp_path)
        with pytest.raises(ConfigError, match="bracketing_blanks"):
            build_thermo_queues(df, _base_config(bracketing_blanks=-1), tmp_path)

    def test_bracketing_blanks_in_sample_key(self, tmp_path):
        df = _sample_df(plates=(1,))
        cfg = _base_config(bracketing_blanks=3)
        outputs = build_thermo_queues(df, cfg, tmp_path)
        key = pd.read_csv(outputs["_key_csv"])
        bracket = key[key["blank_kind"] == "bracketing"]
        # 2 boxes x (3 lead + 3 trail) = 12
        assert len(bracket) == 12
        assert bracket["is_empty"].all()
        # box_row_index 0-5 per box file (since real samples are 4-7 and 3-6)

    def test_interspersed_blanks_count_and_positions(self, tmp_path):
        # Build a larger fixture so interspersed blanks actually spread.
        wells = {"B2": [f"B{c}" for c in range(2, 23, 2)]}  # 11 real B2 wells
        df = _sample_df(wells_per_plate=wells)
        cfg = _base_config(interspersed_blanks=4)
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        assert len(b2) == 11 + 4  # real + interspersed
        blanks = b2[b2["File Name"].str.contains("BLANK")]
        assert len(blanks) == 4
        # Interspersed wells from pool: H4..H9 first (since n_bracket=0)
        assert set(blanks["Position"]) <= {f"S2:H{i}" for i in range(4, 10)} | {
            f"S2:{r}12" for r in "ABCDEFG"
        }

    def test_interspersed_blanks_skip_bracketing_wells(self, tmp_path):
        """Interspersed pool excludes H wells used by bracketing."""
        wells = {"B2": [f"B{c}" for c in range(2, 23, 2)]}
        df = _sample_df(wells_per_plate=wells)
        cfg = _base_config(bracketing_blanks=3, interspersed_blanks=4)
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        # 11 real + 4 interspersed + 6 bracketing = 21
        assert len(b2) == 21
        # Extract interspersed (not lead/trail): positions in H4..H9 or col 12
        interspersed = b2.iloc[3:-3]  # drop 3 lead + 3 trail
        blanks = interspersed[interspersed["File Name"].str.contains("BLANK")]
        for pos in blanks["Position"]:
            well = pos.split(":")[1]
            assert well in {f"H{i}" for i in range(4, 10)} | {
                f"{r}12" for r in "ABCDEFG"
            }, f"interspersed used reserved bracket well: {pos}"

    def test_interspersed_blanks_pool_exhausted_raises(self, tmp_path):
        df = _sample_df(plates=(1,))
        # Pool size = 6 (H middle) + 7 (col 12) = 13; ask for 14 -> error
        cfg = _base_config(interspersed_blanks=14)
        with pytest.raises(ConfigError, match="exceeds available pool"):
            build_thermo_queues(df, cfg, tmp_path)

    def test_interspersed_blanks_in_key(self, tmp_path):
        df = _sample_df(plates=(1,))
        cfg = _base_config(interspersed_blanks=2)
        outputs = build_thermo_queues(df, cfg, tmp_path)
        key = pd.read_csv(outputs["_key_csv"])
        interspersed = key[key["blank_kind"] == "interspersed"]
        # 2 boxes x 2 interspersed = 4 total
        assert len(interspersed) == 4
        assert interspersed["is_empty"].all()

    def test_group_by_sorts_and_inserts_separators(self, tmp_path):
        # Build a fixture with samples at varying "shapes" counts.
        rows = []
        for well, shapes in [
            ("B2", 1),
            ("B4", 200),
            ("B6", 10),
            ("B8", 1),
            ("B10", 200),
            ("B12", 10),
        ]:
            rows.append(
                {
                    "plate": 1,
                    "well": well,
                    "slide": f"s{well}",
                    "bone": "x",
                    "replicate": 1,
                    "shapes": shapes,
                }
            )
        df = pd.DataFrame(rows)
        cfg = _base_config(
            group_by_column="shapes",
            group_by_ascending=True,
            group_separator_blanks=True,
            shuffle=False,  # easier to assert order
        )
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        # Groups (ascending): 1 (2 samples), 10 (2), 200 (2) -> 2 separators.
        # Total rows: 6 real + 2 separators = 8.
        assert len(b2) == 8
        # Expected order: 2 shape-1 samples, SEP, 2 shape-10, SEP, 2 shape-200
        is_blank = b2["File Name"].str.contains("BLANK")
        assert list(is_blank) == [False, False, True, False, False, True, False, False]

    def test_group_by_with_interspersed_and_bracketing(self, tmp_path):
        rows = []
        shapes_plan = [
            (w, s)
            for w, s in [("B2", 1), ("B4", 1), ("B6", 10), ("B8", 10), ("B10", 100), ("B12", 100)]
        ]
        for well, shapes in shapes_plan:
            rows.append(
                {
                    "plate": 1,
                    "well": well,
                    "slide": f"s{well}",
                    "bone": "x",
                    "replicate": 1,
                    "shapes": shapes,
                }
            )
        df = pd.DataFrame(rows)
        cfg = _base_config(
            group_by_column="shapes",
            group_by_ascending=True,
            group_separator_blanks=True,
            bracketing_blanks=3,
            interspersed_blanks=2,
        )
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        # 6 real + 2 separators + 2 interspersed + 6 bracketing = 16
        assert len(b2) == 16
        key = pd.read_csv(outputs["_key_csv"])
        assert (key["blank_kind"] == "group_separator").sum() == 2
        assert (key["blank_kind"] == "interspersed").sum() == 2
        assert (key["blank_kind"] == "bracketing").sum() == 6

    def test_column_substitutions_apply_to_template(self, tmp_path):
        df = _sample_df(plates=(1,))
        # All slide values start with "slide1" — strip that prefix.
        cfg = _base_config(
            file_name_template="{slide}_{well_384}",
            column_substitutions={"slide": (r"^slide1", "")},
        )
        outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        # Original slide values were like "slide1B20", "slide1B21", etc.
        # After strip: "B20", "B21", ...
        assert not any(n.startswith("slide1") for n in b2["File Name"])

    def test_column_substitutions_missing_column_raises(self, tmp_path):
        df = _sample_df(plates=(1,))
        cfg = _base_config(column_substitutions={"nosuchcol": (r"^x", "")})
        with pytest.raises(ConfigError, match="nosuchcol"):
            build_thermo_queues(df, cfg, tmp_path)

    def test_forbidden_chars_warn_but_dont_mutate(self, tmp_path, caplog):
        df = _sample_df(plates=(1,))
        # Template with literal colon (forbidden in Windows file names).
        cfg = _base_config(file_name_template="bad:{slide}_{well_384}_{well_96}")
        with caplog.at_level(logging.WARNING):
            outputs = build_thermo_queues(df, cfg, tmp_path)
        b2 = pd.read_csv(outputs["B2"], skiprows=1)
        assert all(":" in n for n in b2["File Name"])  # not mutated
        assert any("reject" in r.message for r in caplog.records)
