"""
Strategy registry for cell detection strategies.

Provides a centralized registry for registering and creating detection strategy
instances. Allows dynamic lookup of strategies by cell type name.

Usage:
    from segmentation.detection.registry import StrategyRegistry

    # Get a strategy instance
    strategy = StrategyRegistry.create('nmj', intensity_percentile=98)

    # List available strategies
    available = StrategyRegistry.list_strategies()
"""

from typing import Dict, Type, List, Any

from .strategies import (
    DetectionStrategy,
    NMJStrategy,
    MKStrategy,
    CellStrategy,
    VesselStrategy,
    MesotheliumStrategy,
    IsletStrategy,
    TissuePatternStrategy,
)


class StrategyRegistry:
    """
    Registry for detection strategy classes.

    A class-based registry (not instantiated) that stores strategy classes
    and provides methods to register, create, and list strategies.

    Attributes:
        _registry: Dict mapping cell type names to strategy classes

    Example:
        # Register a custom strategy
        StrategyRegistry.register('custom', MyCustomStrategy)

        # Create an instance with parameters
        strategy = StrategyRegistry.create('nmj', intensity_percentile=98)

        # List all registered strategies
        strategies = StrategyRegistry.list_strategies()  # ['nmj', 'mk', 'cell', ...]
    """

    _registry: Dict[str, Type[DetectionStrategy]] = {}

    @classmethod
    def register(cls, cell_type: str, strategy_class: Type[DetectionStrategy]) -> None:
        """
        Register a strategy class for a given cell type.

        Args:
            cell_type: Name of the cell type (e.g., 'nmj', 'mk', 'vessel')
            strategy_class: Strategy class (subclass of DetectionStrategy)

        Raises:
            TypeError: If strategy_class is not a subclass of DetectionStrategy
        """
        if not isinstance(strategy_class, type) or not issubclass(strategy_class, DetectionStrategy):
            raise TypeError(
                f"strategy_class must be a subclass of DetectionStrategy, "
                f"got {type(strategy_class)}"
            )
        cls._registry[cell_type] = strategy_class

    @classmethod
    def create(cls, cell_type: str, **kwargs: Any) -> DetectionStrategy:
        """
        Create an instance of a registered strategy.

        Args:
            cell_type: Name of the cell type to create strategy for
            **kwargs: Arguments passed to the strategy constructor

        Returns:
            Instance of the requested strategy class

        Raises:
            KeyError: If cell_type is not registered
        """
        if cell_type not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise KeyError(
                f"Unknown cell type '{cell_type}'. "
                f"Available strategies: {available}"
            )
        strategy_class = cls._registry[cell_type]
        return strategy_class(**kwargs)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all registered strategy names.

        Returns:
            List of registered cell type names
        """
        return list(cls._registry.keys())

    @classmethod
    def get_strategy_class(cls, cell_type: str) -> Type[DetectionStrategy]:
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
            available = ', '.join(cls._registry.keys())
            raise KeyError(
                f"Unknown cell type '{cell_type}'. "
                f"Available strategies: {available}"
            )
        return cls._registry[cell_type]


# Auto-register existing strategies
StrategyRegistry.register('nmj', NMJStrategy)
StrategyRegistry.register('mk', MKStrategy)
StrategyRegistry.register('cell', CellStrategy)
StrategyRegistry.register('vessel', VesselStrategy)
StrategyRegistry.register('mesothelium', MesotheliumStrategy)
StrategyRegistry.register('islet', IsletStrategy)
StrategyRegistry.register('tissue_pattern', TissuePatternStrategy)
