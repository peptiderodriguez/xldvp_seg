"""
Model registry: metadata catalog for feature extraction and segmentation models.

Tracks available models with their properties (feature dimensions, modality,
license, HuggingFace URL). Does NOT handle model loading -- that stays in
ModelManager (xldvp_seg/models/manager.py). This is a metadata catalog
for discovery and documentation.

Usage:
    from xldvp_seg.models.registry import list_models, get_model_info, ModelRegistry

    # List all models
    for m in list_models():
        print(m.name, m.feature_dim, m.modality)

    # Filter by modality
    brightfield_models = list_models(modality="brightfield")

    # Get specific model info
    info = get_model_info("sam2")
    print(info.description, info.license)

    # Print formatted table
    ModelRegistry.print_models()
"""

from dataclasses import dataclass

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMeta:
    """Metadata for a registered model."""

    name: str
    task: str  # "embedding", "segmentation", "multimodal"
    feature_dim: int  # output embedding dimension (0 for segmentation-only)
    modality: str  # "fluorescence", "brightfield", "both"
    description: str = ""
    license: str = ""
    input_size: int = 224
    hf_url: str = ""
    auto_download: bool = False
    gated: bool = False  # requires HuggingFace token for download
    installed: bool = False


class ModelRegistry:
    """
    Registry for model metadata.

    A class-based registry (not instantiated) that stores model metadata
    and provides methods to register, look up, filter, and display models.

    This is purely a metadata catalog. Model loading and lifecycle management
    are handled by ModelManager in manager.py.
    """

    _registry: dict[str, ModelMeta] = {}

    @classmethod
    def reset(cls):
        """Reset the registry to its default state (for testing)."""
        cls._registry.clear()
        _register_defaults()

    @classmethod
    def register(cls, meta: ModelMeta) -> None:
        """
        Register a model's metadata.

        Args:
            meta: ModelMeta instance with model properties
        """
        if meta.name in cls._registry:
            logger.warning("Overwriting model '%s'", meta.name)
        cls._registry[meta.name] = meta
        logger.debug(
            "Registered model: %s (task=%s, dim=%d)", meta.name, meta.task, meta.feature_dim
        )

    @classmethod
    def get(cls, name: str) -> ModelMeta:
        """
        Get metadata for a registered model.

        Args:
            name: Model name

        Returns:
            ModelMeta instance

        Raises:
            KeyError: If model is not registered
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(f"Model '{name}' not registered. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list_models(
        cls,
        modality: str | None = None,
        task: str | None = None,
    ) -> list[ModelMeta]:
        """
        List registered models, optionally filtered by modality and/or task.

        Args:
            modality: Filter to models supporting this modality.
                      Models with modality="both" match any filter value.
            task: Filter to models with this task type.

        Returns:
            Sorted list of ModelMeta instances
        """
        models = list(cls._registry.values())
        if modality:
            models = [m for m in models if m.modality in (modality, "both")]
        if task:
            models = [m for m in models if m.task == task]
        return sorted(models, key=lambda m: m.name)

    @classmethod
    def print_models(cls, modality: str | None = None) -> None:
        """Print a formatted table of registered models."""
        models = cls.list_models(modality=modality)
        if not models:
            print("No models registered.")
            return
        print(f"{'Name':<18} {'Task':<14} {'Dim':<6} {'Modality':<14} {'Status':<12} License")
        print("-" * 82)
        for m in models:
            if m.installed:
                status = "installed"
            elif m.gated:
                status = "gated (HF)"
            else:
                status = "available"
            print(
                f"{m.name:<18} {m.task:<14} {m.feature_dim:<6d} "
                f"{m.modality:<14} {status:<12} {m.license}"
            )


def register_model(name, task, feature_dim, modality, **kwargs):
    """
    Convenience function to register a model.

    Args:
        name: Model identifier
        task: "embedding", "segmentation", or "multimodal"
        feature_dim: Output embedding dimension (0 for segmentation-only)
        modality: "fluorescence", "brightfield", or "both"
        **kwargs: Additional ModelMeta fields (description, license, hf_url, etc.)
    """
    ModelRegistry.register(
        ModelMeta(name=name, task=task, feature_dim=feature_dim, modality=modality, **kwargs)
    )


def list_models(modality=None, task=None):
    """List registered models, optionally filtered. See ModelRegistry.list_models()."""
    return ModelRegistry.list_models(modality=modality, task=task)


def get_model_info(name):
    """Get metadata for a model by name. See ModelRegistry.get()."""
    return ModelRegistry.get(name)


# ---------------------------------------------------------------------------
# Register known models at module load time
# ---------------------------------------------------------------------------


def _register_defaults():
    """Register all built-in model metadata."""

    # --- Existing models (installed in the pipeline) ---

    register_model(
        "sam2",
        "embedding",
        256,
        "both",
        description="SAM2 spatial point embeddings",
        license="Apache-2.0",
        installed=True,
    )

    register_model(
        "resnet50",
        "embedding",
        4096,
        "both",
        description="ResNet-50 ImageNet features (2x2048D: masked + context)",
        license="BSD-3",
        installed=True,
    )

    register_model(
        "dinov2_vitl14",
        "embedding",
        2048,
        "both",
        description="DINOv2 ViT-L/14 features (2x1024D: masked + context)",
        license="Apache-2.0",
        installed=True,
    )

    register_model(
        "cellpose",
        "segmentation",
        0,
        "both",
        description="Cellpose cell segmentation (cpsam model)",
        license="BSD-3",
        installed=True,
    )

    # --- Brightfield foundation models (gated, require HuggingFace token) ---
    # Download with: xlseg download-models --brightfield
    # /analyze walks novice users through HF account + token setup.

    register_model(
        "uni2",
        "embedding",
        1536,
        "brightfield",
        description="UNI2 ViT-Giant/14 pathology FM (Mahmood Lab)",
        license="CC-BY-NC-ND",
        hf_url="MahmoodLab/UNI2-h",
        auto_download=True,
        gated=True,
    )

    register_model(
        "virchow2",
        "embedding",
        2560,
        "brightfield",
        description="Virchow2 pathology FM, mixed magnification (Paige AI)",
        license="CC-BY-NC-ND",
        hf_url="paige-ai/Virchow2",
        auto_download=True,
        gated=True,
    )

    register_model(
        "conch",
        "multimodal",
        512,
        "brightfield",
        description="CONCH multimodal image+text pathology model (Mahmood Lab)",
        license="CC-BY-NC-ND",
        hf_url="MahmoodLab/CONCH",
        auto_download=True,
        gated=True,
    )

    register_model(
        "phikon_v2",
        "embedding",
        1024,
        "brightfield",
        description="Phikon-v2 ViT-L/16 DINOv2-based pathology FM (Owkin)",
        license="Non-commercial",
        hf_url="owkin/phikon-v2",
        auto_download=True,
        gated=True,
    )

    register_model(
        "h_optimus_1",
        "embedding",
        1536,
        "brightfield",
        description="H-optimus-1 ViT-G/14 pathology FM (Bioptimus)",
        license="Apache-2.0",
        hf_url="bioptimus/H-optimus-1",
        auto_download=True,
        gated=False,
    )

    # --- Planned alternative segmenters ---

    register_model(
        "instanseg",
        "segmentation",
        0,
        "both",
        description="InstanSeg lightweight instance segmentation (3.8M params)",
        license="Apache-2.0",
        auto_download=True,
    )


_register_defaults()
