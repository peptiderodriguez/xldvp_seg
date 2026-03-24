"""Tests for strategy and model registries."""

import pytest


class TestStrategyRegistry:
    """Test the @register_strategy decorator and StrategyRegistry."""

    def test_all_strategies_registered(self):
        import segmentation.detection.strategies
        from segmentation.detection.registry import StrategyRegistry
        names = StrategyRegistry.list_strategies()
        for expected in ["cell", "mk", "nmj", "vessel", "islet",
                         "mesothelium", "tissue_pattern", "instanseg"]:
            assert expected in names, f"Strategy '{expected}' not registered"

    def test_get_strategy_class_returns_type(self):
        import segmentation.detection.strategies
        from segmentation.detection.registry import StrategyRegistry
        cls = StrategyRegistry.get_strategy_class("cell")
        assert isinstance(cls, type)

    def test_get_metadata_has_description(self):
        import segmentation.detection.strategies
        from segmentation.detection.registry import StrategyRegistry
        for name in StrategyRegistry.list_strategies():
            meta = StrategyRegistry.get_metadata(name)
            assert meta.description, f"Strategy '{name}' has empty description"

    def test_unknown_strategy_raises(self):
        from segmentation.detection.registry import StrategyRegistry
        with pytest.raises(KeyError, match="Unknown cell type"):
            StrategyRegistry.get_strategy_class("nonexistent_type")

    def test_print_strategies_no_error(self, capsys):
        import segmentation.detection.strategies
        from segmentation.detection.registry import StrategyRegistry
        StrategyRegistry.print_strategies()
        captured = capsys.readouterr()
        assert "cell" in captured.out


class TestModelRegistry:
    """Test the model metadata registry."""

    def test_known_models_registered(self):
        from segmentation.models.registry import list_models
        names = {m.name for m in list_models()}
        for expected in ["sam2", "resnet50", "dinov2_vitl14", "cellpose"]:
            assert expected in names, f"Model '{expected}' not registered"

    def test_brightfield_models_registered(self):
        from segmentation.models.registry import list_models
        names = {m.name for m in list_models()}
        for expected in ["uni2", "virchow2", "conch", "phikon_v2"]:
            assert expected in names, f"Brightfield model '{expected}' not registered"

    def test_filter_by_modality(self):
        from segmentation.models.registry import list_models
        bf_models = list_models(modality="brightfield")
        for m in bf_models:
            assert m.modality in ("brightfield", "both")

    def test_gated_models_have_hf_url(self):
        from segmentation.models.registry import list_models
        for m in list_models():
            if m.gated:
                assert m.hf_url, f"Gated model '{m.name}' missing hf_url"

    def test_feature_dims_positive(self):
        from segmentation.models.registry import list_models
        for m in list_models():
            if m.task == "embedding":
                assert m.feature_dim > 0, f"Model '{m.name}' has feature_dim={m.feature_dim}"

    def test_print_models_no_error(self, capsys):
        from segmentation.models.registry import ModelRegistry
        ModelRegistry.print_models()
        captured = capsys.readouterr()
        assert "sam2" in captured.out
