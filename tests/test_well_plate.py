"""Tests for segmentation.lmd.well_plate — 384-well plate serpentine generation."""

import pytest
import re

from segmentation.lmd.well_plate import (
    generate_quadrant_serpentine,
    generate_plate_wells,
    generate_multiplate_wells,
    insert_empty_wells,
    WELLS_PER_PLATE,
)


class TestQuadrantSerpentine:
    def test_b2_generates_77_wells(self):
        wells = generate_quadrant_serpentine("B2")
        assert len(wells) == 77

    def test_all_quadrants_77_wells(self):
        for q in ["B2", "B3", "C2", "C3"]:
            wells = generate_quadrant_serpentine(q)
            assert len(wells) == 77, f"Quadrant {q} has {len(wells)} wells"

    def test_well_format(self):
        wells = generate_quadrant_serpentine("B2")
        for w in wells:
            assert re.match(r"^[A-P]\d{1,2}$", w), f"Invalid well: {w}"

    def test_no_duplicates(self):
        wells = generate_quadrant_serpentine("B2")
        assert len(set(wells)) == len(wells)

    def test_invalid_quadrant_raises(self):
        with pytest.raises(ValueError):
            generate_quadrant_serpentine("A1")

    def test_start_corners(self):
        """All four start corners should produce 77 unique wells."""
        for corner in ["TL", "TR", "BL", "BR"]:
            wells = generate_quadrant_serpentine("B2", start_corner=corner)
            assert len(wells) == 77
            assert len(set(wells)) == 77

    def test_invalid_start_corner_raises(self):
        with pytest.raises(ValueError):
            generate_quadrant_serpentine("B2", start_corner="XX")

    def test_b2_uses_even_rows_even_cols(self):
        """B2 quadrant should only contain even rows (B,D,F,H,J,L,N) and even cols."""
        wells = generate_quadrant_serpentine("B2")
        even_rows = set("BDFHJLN")
        even_cols = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}
        for w in wells:
            row = w[0]
            col = int(w[1:])
            assert row in even_rows, f"Well {w} has unexpected row {row}"
            assert col in even_cols, f"Well {w} has unexpected col {col}"

    def test_c3_uses_odd_rows_odd_cols(self):
        """C3 quadrant should only contain odd rows (C,E,G,I,K,M,O) and odd cols."""
        wells = generate_quadrant_serpentine("C3")
        odd_rows = set("CEGIKMO")
        odd_cols = {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}
        for w in wells:
            row = w[0]
            col = int(w[1:])
            assert row in odd_rows, f"Well {w} has unexpected row {row}"
            assert col in odd_cols, f"Well {w} has unexpected col {col}"


class TestPlateWells:
    def test_generates_308_wells(self):
        wells = generate_plate_wells(308)
        assert len(wells) == 308

    def test_no_duplicates(self):
        wells = generate_plate_wells(308)
        assert len(set(wells)) == len(wells)

    def test_wells_per_plate_constant(self):
        assert WELLS_PER_PLATE == 308

    def test_partial_plate(self):
        wells = generate_plate_wells(100)
        assert len(wells) == 100

    def test_zero_wells(self):
        wells = generate_plate_wells(0)
        assert wells == []

    def test_exceeds_plate_raises(self):
        with pytest.raises(ValueError):
            generate_plate_wells(309)


class TestMultiplateWells:
    def test_single_plate(self):
        result = generate_multiplate_wells(200)
        assert len(result) == 200
        # All on plate 1
        plates = {plate for plate, _ in result}
        assert plates == {1}

    def test_overflow_to_second_plate(self):
        result = generate_multiplate_wells(400)
        assert len(result) == 400
        plates = {plate for plate, _ in result}
        assert plates == {1, 2}

    def test_exact_plate_boundary(self):
        result = generate_multiplate_wells(308)
        assert len(result) == 308
        plates = {plate for plate, _ in result}
        assert plates == {1}

    def test_zero_wells(self):
        result = generate_multiplate_wells(0)
        assert result == []

    def test_return_format(self):
        """Each element should be (plate_number, well_address) tuple."""
        result = generate_multiplate_wells(10)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            plate, well = item
            assert isinstance(plate, int)
            assert isinstance(well, str)


class TestInsertEmptyWells:
    def test_empties_inserted(self):
        wells = generate_plate_wells(120)
        n_samples = 100
        result_wells, empty_positions = insert_empty_wells(wells, n_samples, empty_pct=10)
        # Result wells should be the same list passed in
        assert len(result_wells) == 120
        # Empty positions should be a set of indices
        assert isinstance(empty_positions, set)
        assert len(empty_positions) > 0

    def test_empty_count_matches_pct(self):
        """Number of empty wells should be ceil(n_samples * empty_pct / 100)."""
        import math

        wells = generate_plate_wells(200)
        n_samples = 100
        _, empty_positions = insert_empty_wells(wells, n_samples, empty_pct=10)
        expected = max(1, math.ceil(n_samples * 10 / 100))
        assert len(empty_positions) == expected

    def test_deterministic_with_seed(self):
        """Same seed should produce same empty positions."""
        wells = generate_plate_wells(200)
        _, pos1 = insert_empty_wells(wells, 100, empty_pct=10, seed=42)
        _, pos2 = insert_empty_wells(wells, 100, empty_pct=10, seed=42)
        assert pos1 == pos2

    def test_different_seeds_differ(self):
        """Different seeds should generally produce different positions."""
        wells = generate_plate_wells(200)
        _, pos1 = insert_empty_wells(wells, 100, empty_pct=10, seed=42)
        _, pos2 = insert_empty_wells(wells, 100, empty_pct=10, seed=99)
        # Very unlikely to be the same with different seeds
        assert pos1 != pos2

    def test_positions_within_bounds(self):
        """All empty positions should be valid indices into the well list."""
        wells = generate_plate_wells(150)
        _, empty_positions = insert_empty_wells(wells, 100, empty_pct=10)
        for pos in empty_positions:
            assert 0 <= pos < len(wells)
