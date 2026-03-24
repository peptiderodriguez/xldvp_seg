"""
Strategy registry for cell detection strategies.

Provides a centralized registry for registering and looking up detection strategy
classes by cell type name. Strategies self-register via the @register_strategy
decorator at import time.

IMPORTANT: This module must NOT import from strategies/ — the strategies import
from here (not the other way around). Registration happens when strategy modules
are imported (triggered by importing segmentation.detection.strategies).

Usage:
    from segmentation.detection.registry import StrategyRegistry, register_strategy

    @register_strategy("mytype", description="My detection method", channels=["ch1"])
    class MyStrategy(DetectionStrategy):
        ...

    # Later, after strategies are imported:
    strategy_class = StrategyRegistry.get_strategy_class("mytype")
    metadata = StrategyRegistry.get_metadata("mytype")
    StrategyRegistry.print_strategies()
"""

from dataclasses import dataclass, field
from typing import Dict, Type, List, Optional

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyMeta:
    """Metadata for a registered detection strategy."""

    name: str
    description: str
    channels: List[str] = field(default_factory=list)
    feature_dims: Optional[int] = None
    strategy_class: Optional[Type] = None


class StrategyRegistry:
    """
    Registry for detection strategy classes.

    A class-based registry (not instantiated) that stores strategy classes
    with their metadata and provides methods to register, look up, and list strategies.

    Registration is done via the @register_strategy decorator on strategy classes,
    triggered at import time when segmentation.detection.strategies is imported.

    Attributes:
        _registry: Dict mapping cell type names to StrategyMeta

    Example:
        # Look up a strategy class
        cls = StrategyRegistry.get_strategy_class('nmj')

        # Get metadata
        meta = StrategyRegistry.get_metadata('nmj')
        print(meta.description, meta.channels)

        # List all registered strategies
        names = StrategyRegistry.list_strategies()  # ['cell', 'islet', 'mk', ...]

        # Print formatted table
        StrategyRegistry.print_strategies()
    """

    _registry: Dict[str, StrategyMeta] = {}

    @classmethod
    def register(
        cls,
        name: str,
        strategy_class: Type,
        *,
        description: str = "",
        channels: Optional[List[str]] = None,
        feature_dims: Optional[int] = None,
    ) -> None:
        """
        Register a strategy class for a given cell type.

        Args:
            name: Name of the cell type (e.g., 'nmj', 'mk', 'vessel')
            strategy_class: Strategy class (subclass of DetectionStrategy)
            description: Human-readable description of the detection method
            channels: List of expected channel names (e.g., ['cyto', 'nuc'])
            feature_dims: Total feature dimensions produced by this strategy

        Raises:
            TypeError: If strategy_class is not a class
        """
        if not isinstance(strategy_class, type):
            raise TypeError(
                f"strategy_class must be a class, got {type(strategy_class)}"
            )
        if name in cls._registry:
            existing = cls._registry[name].strategy_class
            if existing is not strategy_class:
                logger.warning(
                    "Overwriting strategy '%s': %s -> %s",
                    name, existing.__name__, strategy_class.__name__,
                )
        meta = StrategyMeta(
            name=name,
            description=description,
            channels=channels or [],
            feature_dims=feature_dims,
            strategy_class=strategy_class,
        )
        cls._registry[name] = meta
        logger.debug("Registered strategy: %s (%s)", name, strategy_class.__name__)

    @classmethod
    def get_strategy_class(cls, cell_type: str) -> Type:
        """
        Get the strategy class for a given cell type without instantiating.

        Args:
            cell_type: Name of the cell type

        Returns:
            Strategy class (not instance)

        Raises:
            KeyError: If cell_type is not registered
        """
        if cell_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Unknown cell type '{cell_type}'. "
                f"Available strategies: {available}"
            )
        return cls._registry[cell_type].strategy_class

    @classmethod
    def get_metadata(cls, cell_type: str) -> StrategyMeta:
        """
        Get full metadata for a registered strategy.

        Args:
            cell_type: Name of the cell type

        Returns:
            StrategyMeta with name, description, channels, feature_dims, strategy_class

        Raises:
            KeyError: If cell_type is not registered
        """
        if cell_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Unknown cell type '{cell_type}'. "
                f"Available strategies: {available}"
            )
        return cls._registry[cell_type]

    @classmethod
    def create(cls, cell_type: str, **kwargs) -> object:
        """
        Create an instance of a registered strategy.

        Kept for backward compatibility with existing code and tests.

        Args:
            cell_type: Name of the cell type to create strategy for
            **kwargs: Arguments passed to the strategy constructor

        Returns:
            Instance of the requested strategy class

        Raises:
            KeyError: If cell_type is not registered
        """
        strategy_class = cls.get_strategy_class(cell_type)
        return strategy_class(**kwargs)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all registered strategy names.

        Returns:
            Sorted list of registered cell type names
        """
        return sorted(cls._registry.keys())

    @classmethod
    def print_strategies(cls) -> None:
        """Print a formatted table of all registered strategies."""
        entries = sorted(cls._registry.values(), key=lambda m: m.name)
        if not entries:
            print("No strategies registered.")
            return
        print(f"{'Name':<20s} {'Channels':<20s} {'Description'}")
        print("-" * 80)
        for meta in entries:
            channels_str = ", ".join(meta.channels) if meta.channels else "(auto)"
            print(f"{meta.name:<20s} {channels_str:<20s} {meta.description}")


def register_strategy(name, *, description="", channels=None, feature_dims=None):
    """
    Decorator to register a detection strategy class.

    Applied to strategy class definitions so they self-register at import time.

    Args:
        name: Cell type name for registry lookup (e.g., 'nmj', 'cell')
        description: Human-readable description of the detection method
        channels: List of expected channel names (e.g., ['cyto', 'nuc'])
        feature_dims: Total feature dimensions produced

    Usage:
        @register_strategy("nmj", description="NMJ detection", channels=["BTX"])
        class NMJStrategy(DetectionStrategy):
            ...
    """

    def decorator(cls):
        StrategyRegistry.register(
            name,
            cls,
            description=description,
            channels=channels or [],
            feature_dims=feature_dims,
        )
        return cls

    return decorator
