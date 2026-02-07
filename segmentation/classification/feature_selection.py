"""
Feature selection utilities for vessel classification.

Provides tools for analyzing feature importance, selecting optimal feature subsets,
and cross-validation utilities for model evaluation.

Usage:
    from segmentation.classification.feature_selection import (
        analyze_feature_importance,
        select_optimal_features,
        cross_validate_features,
    )

    # Analyze importance
    importance = analyze_feature_importance(X, y, feature_names)

    # Select optimal subset
    optimal_features = select_optimal_features(X, y, feature_names, method='rfecv')

    # Cross-validate with specific features
    scores = cross_validate_features(X, y, feature_subset)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    mutual_info_classif,
)
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def analyze_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = 'all',
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze feature importance using multiple methods.

    Methods:
    - 'gini': Random Forest Gini importance (default)
    - 'permutation': Permutation importance (more reliable but slower)
    - 'mutual_info': Mutual information with target
    - 'all': Run all methods and aggregate

    Args:
        X: Feature matrix (N, D)
        y: Labels (N,)
        feature_names: Names of features
        method: Importance method ('gini', 'permutation', 'mutual_info', 'all')
        n_repeats: Number of repeats for permutation importance
        random_state: Random seed

    Returns:
        Dictionary with importance scores per method:
        {
            'gini': {'feature1': 0.15, 'feature2': 0.12, ...},
            'permutation': {...},
            'mutual_info': {...},
            'aggregate': {...}  # Average rank across methods
        }
    """
    # Prepare data
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels if strings
    if y.dtype.kind in ('U', 'S', 'O'):
        le = LabelEncoder()
        y = le.fit_transform(y)

    results = {}

    # Train base model
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_scaled, y)

    # Gini importance (MDI - Mean Decrease in Impurity)
    if method in ('gini', 'all'):
        gini_importance = rf.feature_importances_
        results['gini'] = dict(zip(feature_names, gini_importance.tolist()))
        logger.info("Computed Gini importance")

    # Permutation importance
    if method in ('permutation', 'all'):
        perm_importance = permutation_importance(
            rf, X_scaled, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        results['permutation'] = dict(zip(
            feature_names,
            perm_importance.importances_mean.tolist()
        ))
        logger.info("Computed permutation importance")

    # Mutual information
    if method in ('mutual_info', 'all'):
        mi_scores = mutual_info_classif(
            X_scaled, y,
            random_state=random_state,
            n_neighbors=5
        )
        results['mutual_info'] = dict(zip(feature_names, mi_scores.tolist()))
        logger.info("Computed mutual information")

    # Aggregate: average rank across methods
    if method == 'all' and len(results) > 1:
        # Convert to ranks (higher importance = lower rank = better)
        ranks = {}
        for method_name, importance_dict in results.items():
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for rank, (feat, _) in enumerate(sorted_features):
                if feat not in ranks:
                    ranks[feat] = []
                ranks[feat].append(rank)

        # Average rank
        aggregate = {feat: np.mean(rank_list) for feat, rank_list in ranks.items()}
        results['aggregate'] = aggregate

        # Also compute normalized aggregate score (inverse rank)
        max_rank = len(feature_names) - 1
        results['aggregate_score'] = {
            feat: (max_rank - avg_rank) / max_rank
            for feat, avg_rank in aggregate.items()
        }

    return results


def select_optimal_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = 'rfecv',
    min_features: int = 5,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select optimal feature subset using various methods.

    Methods:
    - 'rfecv': Recursive Feature Elimination with CV (recommended)
    - 'rfe': RFE without CV, select top N features
    - 'threshold': Select features above importance threshold
    - 'top_n': Select top N features by importance

    Args:
        X: Feature matrix (N, D)
        y: Labels (N,)
        feature_names: Names of features
        method: Selection method
        min_features: Minimum number of features to keep
        cv_folds: Number of CV folds for RFECV
        random_state: Random seed
        verbose: Print progress

    Returns:
        Dictionary with:
        - 'selected_features': List of selected feature names
        - 'selected_indices': Indices of selected features
        - 'n_features': Number of selected features
        - 'cv_scores': CV scores (if applicable)
        - 'ranking': Feature ranking
    """
    # Prepare data
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    if y.dtype.kind in ('U', 'S', 'O'):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Base estimator
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    result = {}

    if method == 'rfecv':
        # Recursive Feature Elimination with Cross-Validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        selector = RFECV(
            estimator=rf,
            step=1,
            cv=cv,
            scoring='accuracy',
            min_features_to_select=min_features,
            n_jobs=-1
        )
        selector.fit(X_scaled, y)

        selected_mask = selector.support_
        result['cv_scores'] = selector.cv_results_['mean_test_score'].tolist()
        result['optimal_n_features'] = selector.n_features_
        result['ranking'] = selector.ranking_.tolist()

        if verbose:
            logger.info(f"RFECV selected {selector.n_features_} features")
            logger.info(f"Best CV score: {max(selector.cv_results_['mean_test_score']):.4f}")

    elif method == 'rfe':
        # RFE without CV
        selector = RFE(
            estimator=rf,
            n_features_to_select=min_features,
            step=1
        )
        selector.fit(X_scaled, y)
        selected_mask = selector.support_
        result['ranking'] = selector.ranking_.tolist()

        if verbose:
            logger.info(f"RFE selected {min_features} features")

    elif method == 'threshold':
        # Select features above mean importance
        rf.fit(X_scaled, y)
        selector = SelectFromModel(rf, prefit=True, threshold='mean')
        selected_mask = selector.get_support()

        # Ensure minimum features
        if selected_mask.sum() < min_features:
            # Fall back to top N
            importance = rf.feature_importances_
            top_indices = np.argsort(importance)[-min_features:]
            selected_mask = np.zeros(len(feature_names), dtype=bool)
            selected_mask[top_indices] = True

        if verbose:
            logger.info(f"Threshold selection: {selected_mask.sum()} features")

    elif method == 'top_n':
        # Simply select top N by importance
        rf.fit(X_scaled, y)
        importance = rf.feature_importances_
        top_indices = np.argsort(importance)[-min_features:]
        selected_mask = np.zeros(len(feature_names), dtype=bool)
        selected_mask[top_indices] = True

        if verbose:
            logger.info(f"Selected top {min_features} features by importance")

    else:
        raise ValueError(f"Unknown selection method: {method}")

    # Extract selected features
    selected_indices = np.where(selected_mask)[0].tolist()
    selected_features = [feature_names[i] for i in selected_indices]

    result.update({
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'n_features': len(selected_features),
        'method': method,
    })

    return result


def cross_validate_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    feature_subset: Optional[List[str]] = None,
    feature_indices: Optional[List[int]] = None,
    cv_folds: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
    return_models: bool = False,
) -> Dict[str, Any]:
    """
    Cross-validate classifier with specified feature subset.

    Useful for comparing performance with different feature sets.

    Args:
        X: Feature matrix (N, D)
        y: Labels (N,)
        feature_names: All feature names (required if using feature_subset)
        feature_subset: List of feature names to use
        feature_indices: Alternative: indices of features to use
        cv_folds: Number of CV folds
        n_estimators: Number of trees
        random_state: Random seed
        return_models: Whether to return trained models

    Returns:
        Dictionary with:
        - 'mean_accuracy': Mean CV accuracy
        - 'std_accuracy': Standard deviation
        - 'fold_scores': Per-fold scores
        - 'models': List of trained models (if return_models=True)
    """
    # Prepare data
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Select feature subset
    if feature_subset is not None and feature_names is not None:
        feature_indices = [feature_names.index(f) for f in feature_subset if f in feature_names]

    if feature_indices is not None:
        X = X[:, feature_indices]

    # Encode labels
    if y.dtype.kind in ('U', 'S', 'O'):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Cross-validation with Pipeline to avoid data leakage
    # (scaler fits inside each fold, not on full dataset)
    from sklearn.pipeline import Pipeline
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )),
    ])

    fold_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')

    result = {
        'mean_accuracy': float(fold_scores.mean()),
        'std_accuracy': float(fold_scores.std()),
        'fold_scores': fold_scores.tolist(),
        'n_features': X.shape[1],
    }

    # Optionally return trained models
    if return_models:
        models = []
        for train_idx, _ in cv.split(X_scaled, y):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            model.fit(X_scaled[train_idx], y[train_idx])
            models.append(model)
        result['models'] = models

    logger.info(f"CV Accuracy: {fold_scores.mean():.4f} (+/- {fold_scores.std() * 2:.4f})")

    return result


def compute_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Compute learning curve to diagnose bias/variance issues.

    Args:
        X: Feature matrix
        y: Labels
        train_sizes: Array of training set sizes to evaluate
        cv_folds: Number of CV folds
        n_estimators: Number of trees
        random_state: Random seed

    Returns:
        Dictionary with:
        - 'train_sizes': Actual training set sizes used
        - 'train_scores_mean': Mean training scores
        - 'train_scores_std': Std of training scores
        - 'test_scores_mean': Mean test (validation) scores
        - 'test_scores_std': Std of test scores
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    # Prepare data
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    if y.dtype.kind in ('U', 'S', 'O'):
        le = LabelEncoder()
        y = le.fit_transform(y)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    train_sizes_abs, train_scores, test_scores = learning_curve(
        rf, X_scaled, y,
        train_sizes=train_sizes,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state
    )

    return {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': train_scores.mean(axis=1),
        'train_scores_std': train_scores.std(axis=1),
        'test_scores_mean': test_scores.mean(axis=1),
        'test_scores_std': test_scores.std(axis=1),
    }


def compare_feature_sets(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    feature_sets: Dict[str, List[str]],
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance across different feature sets.

    Useful for comparing vessel-only features vs full morphological features.

    Args:
        X: Full feature matrix
        y: Labels
        feature_names: All feature names
        feature_sets: Dict mapping set name to list of feature names
            e.g., {'vessel_only': [...], 'all_features': [...]}
        cv_folds: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary mapping set name to performance metrics
    """
    results = {}

    for set_name, feature_list in feature_sets.items():
        logger.info(f"Evaluating feature set: {set_name} ({len(feature_list)} features)")

        cv_result = cross_validate_features(
            X, y,
            feature_names=feature_names,
            feature_subset=feature_list,
            cv_folds=cv_folds,
            random_state=random_state
        )

        results[set_name] = {
            'mean_accuracy': cv_result['mean_accuracy'],
            'std_accuracy': cv_result['std_accuracy'],
            'n_features': cv_result['n_features'],
        }

    # Print comparison
    logger.info("\n" + "=" * 50)
    logger.info("Feature Set Comparison")
    logger.info("=" * 50)
    for set_name, metrics in sorted(results.items(), key=lambda x: -x[1]['mean_accuracy']):
        logger.info(
            f"{set_name:20s}: {metrics['mean_accuracy']:.4f} "
            f"(+/- {metrics['std_accuracy'] * 2:.4f}) "
            f"[{metrics['n_features']} features]"
        )

    return results


def get_correlated_features(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.9
) -> List[Tuple[str, str, float]]:
    """
    Find highly correlated feature pairs.

    Useful for identifying redundant features that could be removed.

    Args:
        X: Feature matrix
        feature_names: Feature names
        threshold: Correlation threshold (default 0.9)

    Returns:
        List of (feature1, feature2, correlation) tuples
    """
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Find pairs above threshold
    correlated_pairs = []
    n_features = len(feature_names)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = abs(corr_matrix[i, j])
            if corr >= threshold:
                correlated_pairs.append((
                    feature_names[i],
                    feature_names[j],
                    float(corr)
                ))

    # Sort by correlation (descending)
    correlated_pairs.sort(key=lambda x: -x[2])

    if correlated_pairs:
        logger.info(f"Found {len(correlated_pairs)} highly correlated pairs (|r| >= {threshold})")

    return correlated_pairs
