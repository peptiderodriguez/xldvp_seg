#!/usr/bin/env python3
"""
Prepare Random Forest training data from vessel annotations.

This script:
1. Loads annotation JSON exported from the HTML annotation interface
2. Loads vessel detection JSON with full feature vectors
3. Extracts features for annotated vessels
4. Outputs train-ready formats:
   - CSV with features for inspection
   - NumPy arrays (X.npy, y.npy) for direct RF training
   - Pickle file with scaler and feature names
   - JSON format compatible with scikit-learn workflows

Usage:
    python scripts/prepare_rf_training_data.py \
        --annotations vessel_annotations.json \
        --detections vessel_detections.json \
        --output-dir /path/to/rf_training_data

    # Or with sklearn JSON export from HTML:
    python scripts/prepare_rf_training_data.py \
        --sklearn-json vessel_rf_sklearn.json \
        --output-dir /path/to/rf_training_data
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# SIZE CLASS DEFINITIONS FOR STRATIFIED SAMPLING
# =============================================================================

# Size class thresholds (in microns) - must match vessel_features.py
SIZE_CLASS_THRESHOLDS = {
    0: (0, 10),        # capillary: 5-10 um
    1: (10, 50),       # arteriole: 10-50 um
    2: (50, 150),      # small_artery: 50-150 um
    3: (150, float('inf'))  # artery: >150 um
}

SIZE_CLASS_NAMES = {
    0: 'capillary',
    1: 'arteriole',
    2: 'small_artery',
    3: 'artery'
}


def get_size_class_from_diameter(diameter_um: float) -> int:
    """Get size class for a vessel based on outer diameter."""
    if diameter_um < 10:
        return 0  # capillary
    elif diameter_um < 50:
        return 1  # arteriole
    elif diameter_um < 150:
        return 2  # small_artery
    else:
        return 3  # artery


def extract_diameters(
    detections: Dict[str, Dict],
    uids: List[str],
) -> np.ndarray:
    """
    Extract outer diameter values for given UIDs.

    Args:
        detections: Dict mapping uid to detection dict
        uids: List of UIDs to extract diameters for

    Returns:
        Array of diameters (NaN for missing values)
    """
    diameters = []
    for uid in uids:
        if uid not in detections:
            diameters.append(np.nan)
            continue

        det = detections[uid]
        features = det.get('features', det)

        # Try different possible field names for diameter
        diameter = features.get('outer_diameter_um')
        if diameter is None:
            diameter = features.get('diameter_um')
        if diameter is None:
            # Fall back to estimating from area
            area = features.get('outer_area_um2') or features.get('area')
            if area is not None and area > 0:
                diameter = 2 * np.sqrt(area / np.pi)

        diameters.append(diameter if diameter is not None else np.nan)

    return np.array(diameters)


def analyze_size_distribution(
    diameters: np.ndarray,
    labels: np.ndarray,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Analyze the size distribution of vessels.

    Args:
        diameters: Array of vessel diameters in microns
        labels: Array of labels (0=negative, 1=positive)
        prefix: Optional prefix for log messages

    Returns:
        Dict with distribution statistics and warnings
    """
    valid_mask = ~np.isnan(diameters)
    valid_diameters = diameters[valid_mask]
    valid_labels = labels[valid_mask]

    if len(valid_diameters) == 0:
        return {'warning': 'No valid diameter measurements found'}

    # Get size classes
    size_classes = np.array([get_size_class_from_diameter(d) for d in valid_diameters])

    # Count per class
    class_counts = Counter(size_classes)
    total = len(size_classes)

    # Calculate percentages
    distribution = {}
    warnings = []

    logger.info(f"\n{prefix}Size Distribution Analysis:")
    logger.info("-" * 50)

    for class_id in range(4):
        count = class_counts.get(class_id, 0)
        pct = 100 * count / total if total > 0 else 0
        class_name = SIZE_CLASS_NAMES[class_id]
        threshold = SIZE_CLASS_THRESHOLDS[class_id]

        distribution[class_name] = {
            'count': count,
            'percentage': pct,
            'threshold_um': threshold,
        }

        # Count positive/negative within class
        class_mask = size_classes == class_id
        if class_mask.sum() > 0:
            pos_in_class = valid_labels[class_mask].sum()
            neg_in_class = class_mask.sum() - pos_in_class
            pos_pct = 100 * pos_in_class / class_mask.sum()
        else:
            pos_in_class = neg_in_class = pos_pct = 0

        logger.info(f"  {class_name:15s}: {count:5d} ({pct:5.1f}%) "
                   f"[pos: {pos_in_class}, neg: {neg_in_class}, pos%: {pos_pct:.1f}%]")

        # Generate warnings for imbalanced distribution
        if pct < 5 and total > 100:
            warnings.append(f"UNDERREPRESENTED: {class_name} has only {pct:.1f}% of samples")
        if pct > 70:
            warnings.append(f"OVERREPRESENTED: {class_name} has {pct:.1f}% of samples")

    # Overall statistics
    logger.info(f"\n  Total valid samples: {total}")
    logger.info(f"  Diameter range: {valid_diameters.min():.1f} - {valid_diameters.max():.1f} um")
    logger.info(f"  Median diameter: {np.median(valid_diameters):.1f} um")

    # Log warnings
    if warnings:
        logger.warning("\n  SIZE DISTRIBUTION WARNINGS:")
        for w in warnings:
            logger.warning(f"    - {w}")

    return {
        'distribution': distribution,
        'warnings': warnings,
        'total': total,
        'diameter_range': (float(valid_diameters.min()), float(valid_diameters.max())),
        'median_diameter': float(np.median(valid_diameters)),
    }


def stratified_sample_by_size(
    X: np.ndarray,
    y: np.ndarray,
    uids: List[str],
    diameters: np.ndarray,
    samples_per_class: Optional[int] = None,
    min_samples_per_class: int = 10,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Perform stratified sampling to balance vessel size classes.

    This ensures the training set has equal representation across size classes,
    preventing the classifier from being biased toward the most common size.

    Args:
        X: Feature matrix
        y: Labels
        uids: Sample UIDs
        diameters: Vessel diameters
        samples_per_class: Target samples per class (None = use min class count)
        min_samples_per_class: Minimum samples to require per class
        random_seed: Random seed

    Returns:
        Tuple of (X_balanced, y_balanced, uids_balanced, diameters_balanced)
    """
    np.random.seed(random_seed)

    valid_mask = ~np.isnan(diameters)
    if not valid_mask.all():
        logger.warning(f"  {(~valid_mask).sum()} samples have missing diameter - will be excluded")

    # Filter to valid samples
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    uids_valid = [uids[i] for i in range(len(uids)) if valid_mask[i]]
    diameters_valid = diameters[valid_mask]

    # Get size classes
    size_classes = np.array([get_size_class_from_diameter(d) for d in diameters_valid])

    # Count per class (separately for positive and negative)
    class_pos_counts = {}
    class_neg_counts = {}
    for class_id in range(4):
        class_mask = size_classes == class_id
        if class_mask.sum() > 0:
            class_pos_counts[class_id] = (y_valid[class_mask] == 1).sum()
            class_neg_counts[class_id] = (y_valid[class_mask] == 0).sum()
        else:
            class_pos_counts[class_id] = 0
            class_neg_counts[class_id] = 0

    # Determine samples per class
    if samples_per_class is None:
        # Use minimum of (smallest positive class, smallest negative class)
        min_pos = min(class_pos_counts.values()) if class_pos_counts else 0
        min_neg = min(class_neg_counts.values()) if class_neg_counts else 0
        samples_per_class = min(min_pos, min_neg)

    if samples_per_class < min_samples_per_class:
        logger.warning(f"  Insufficient samples for balanced stratification. "
                      f"Min class has {samples_per_class} samples, need {min_samples_per_class}")
        logger.warning("  Using class weights instead of undersampling")
        return X_valid, y_valid, uids_valid, diameters_valid

    logger.info(f"\n  Stratified sampling: {samples_per_class} samples per class")

    # Sample from each class
    selected_indices = []

    for class_id in range(4):
        class_mask = size_classes == class_id
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            logger.warning(f"    {SIZE_CLASS_NAMES[class_id]}: NO SAMPLES")
            continue

        # Sample positive and negative separately within each class
        pos_indices = class_indices[y_valid[class_indices] == 1]
        neg_indices = class_indices[y_valid[class_indices] == 0]

        # Sample half from positive, half from negative (or proportional if imbalanced)
        n_pos = min(len(pos_indices), samples_per_class // 2)
        n_neg = min(len(neg_indices), samples_per_class - n_pos)

        if n_pos < samples_per_class // 2:
            # Try to take more from negative to compensate
            n_neg = min(len(neg_indices), samples_per_class - n_pos)

        if n_pos > 0:
            sampled_pos = np.random.choice(pos_indices, n_pos, replace=False)
            selected_indices.extend(sampled_pos.tolist())

        if n_neg > 0:
            sampled_neg = np.random.choice(neg_indices, n_neg, replace=False)
            selected_indices.extend(sampled_neg.tolist())

        logger.info(f"    {SIZE_CLASS_NAMES[class_id]}: {n_pos} pos, {n_neg} neg "
                   f"(of {len(pos_indices)} pos, {len(neg_indices)} neg available)")

    # Create balanced dataset
    selected_indices = np.array(selected_indices)
    X_balanced = X_valid[selected_indices]
    y_balanced = y_valid[selected_indices]
    uids_balanced = [uids_valid[i] for i in selected_indices]
    diameters_balanced = diameters_valid[selected_indices]

    logger.info(f"\n  Balanced dataset: {len(X_balanced)} samples "
               f"({(y_balanced == 1).sum()} pos, {(y_balanced == 0).sum()} neg)")

    return X_balanced, y_balanced, uids_balanced, diameters_balanced


def load_annotations(annotations_path: Path) -> Tuple[set, set, set]:
    """
    Load annotations from JSON file.

    Supports multiple formats:
    - Standard format: {"positive": [...], "negative": [...], "unsure": [...]}
    - sklearn format: {"X": [...], "y": [...], "uids": [...]}

    Returns:
        Tuple of (positive_ids, negative_ids, unsure_ids)
    """
    with open(annotations_path) as f:
        data = json.load(f)

    # Standard format
    if 'positive' in data:
        positive_ids = set(data.get('positive', []))
        negative_ids = set(data.get('negative', []))
        unsure_ids = set(data.get('unsure', []))
        logger.info(f"Loaded annotations: {len(positive_ids)} positive, "
                   f"{len(negative_ids)} negative, {len(unsure_ids)} unsure")
        return positive_ids, negative_ids, unsure_ids

    # sklearn format (already has X, y)
    if 'X' in data and 'y' in data:
        uids = data.get('uids', [])
        y = data['y']
        positive_ids = set(uid for uid, label in zip(uids, y) if label == 1)
        negative_ids = set(uid for uid, label in zip(uids, y) if label == 0)
        logger.info(f"Loaded sklearn format: {len(positive_ids)} positive, "
                   f"{len(negative_ids)} negative")
        return positive_ids, negative_ids, set()

    raise ValueError(f"Unknown annotation format in {annotations_path}")


def load_detections(detections_path: Path) -> Dict[str, Dict]:
    """
    Load vessel detections with features from JSON.

    Supports multiple formats:
    - List of detection dicts with 'uid' or 'id' key
    - Dict keyed by uid

    Returns:
        Dict mapping uid to detection dict
    """
    with open(detections_path) as f:
        data = json.load(f)

    # If already a dict keyed by uid
    if isinstance(data, dict) and not any(k in data for k in ['detections', 'vessels']):
        logger.info(f"Loaded {len(data)} detections (dict format)")
        return data

    # List format
    if isinstance(data, list):
        indexed = {}
        for d in data:
            uid = d.get('uid') or d.get('id')
            if uid:
                indexed[uid] = d
        logger.info(f"Loaded {len(indexed)} detections (list format)")
        return indexed

    # Nested format with 'detections' or 'vessels' key
    detections = data.get('detections') or data.get('vessels') or []
    indexed = {}
    for d in detections:
        uid = d.get('uid') or d.get('id')
        if uid:
            indexed[uid] = d

    logger.info(f"Loaded {len(indexed)} detections")
    return indexed


def extract_feature_matrix(
    detections: Dict[str, Dict],
    uids: List[str],
    feature_names: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract feature matrix for given UIDs.

    Args:
        detections: Dict mapping uid to detection dict
        uids: List of uids to extract features for
        feature_names: Optional list of feature names to use (auto-detect if None)
        exclude_prefixes: Prefixes to exclude (e.g., ['sam2_', 'resnet_'] for morph only)

    Returns:
        Tuple of (X array, valid_uids, feature_names)
    """
    if not uids:
        return np.array([]), [], []

    if exclude_prefixes is None:
        exclude_prefixes = []

    # Auto-detect feature names from first detection
    if feature_names is None:
        sample_uid = next(iter(detections.keys()))
        sample = detections[sample_uid]
        features = sample.get('features', sample)  # Handle both formats

        feature_names = []
        for k, v in sorted(features.items()):
            # Skip non-numeric features
            if not isinstance(v, (int, float, np.integer, np.floating)):
                continue
            # Skip excluded prefixes
            if any(k.startswith(prefix) for prefix in exclude_prefixes):
                continue
            feature_names.append(k)

    X = []
    valid_uids = []

    for uid in uids:
        if uid not in detections:
            logger.debug(f"UID not found in detections: {uid}")
            continue

        det = detections[uid]
        features = det.get('features', det)

        row = []
        for fn in feature_names:
            val = features.get(fn, 0)
            # Handle non-scalar values
            if isinstance(val, (list, tuple)):
                val = 0
            elif val is None:
                val = 0
            row.append(float(val))

        X.append(row)
        valid_uids.append(uid)

    return np.array(X, dtype=np.float32), valid_uids, feature_names


def prepare_training_data(
    annotations_path: Optional[str] = None,
    detections_path: Optional[str] = None,
    sklearn_json_path: Optional[str] = None,
    output_dir: str = './rf_training_data',
    morph_only: bool = False,
    include_sam2: bool = True,
    include_resnet: bool = True,
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify_by_size: bool = False,
    samples_per_size_class: Optional[int] = None,
):
    """
    Prepare training data for Random Forest classifier.

    Args:
        annotations_path: Path to annotation JSON (positive/negative lists)
        detections_path: Path to detection JSON with features
        sklearn_json_path: Path to sklearn-format JSON (alternative input)
        output_dir: Output directory for training files
        morph_only: Use only morphological features (exclude embeddings)
        include_sam2: Include SAM2 embeddings (ignored if morph_only)
        include_resnet: Include ResNet features (ignored if morph_only)
        test_size: Fraction for test split
        random_seed: Random seed for reproducibility
        stratify_by_size: If True, balance training data across vessel size classes
        samples_per_size_class: Target samples per size class (None = auto)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("PREPARING RF TRAINING DATA")
    logger.info("="*70)

    # Determine exclude prefixes based on options
    exclude_prefixes = []
    if morph_only:
        exclude_prefixes = ['sam2_', 'resnet_']
        logger.info("Mode: Morphological features only")
    else:
        if not include_sam2:
            exclude_prefixes.append('sam2_')
        if not include_resnet:
            exclude_prefixes.append('resnet_')

    # Load data based on input format
    if sklearn_json_path:
        # Load from sklearn JSON export (already has X, y)
        logger.info(f"\nLoading sklearn JSON: {sklearn_json_path}")
        with open(sklearn_json_path) as f:
            data = json.load(f)

        X = np.array(data['X'], dtype=np.float32)
        y = np.array(data['y'], dtype=np.int32)
        feature_names = data.get('feature_names', [f'f{i}' for i in range(X.shape[1])])
        uids = data.get('uids', [f'sample_{i}' for i in range(len(y))])

        # Apply feature filtering if needed
        if exclude_prefixes:
            keep_indices = [
                i for i, fn in enumerate(feature_names)
                if not any(fn.startswith(p) for p in exclude_prefixes)
            ]
            X = X[:, keep_indices]
            feature_names = [feature_names[i] for i in keep_indices]
            logger.info(f"Filtered to {len(feature_names)} features")

    else:
        # Load from separate annotation + detection files
        if not annotations_path or not detections_path:
            raise ValueError("Must provide either sklearn_json_path OR both "
                           "annotations_path and detections_path")

        logger.info(f"\nLoading annotations: {annotations_path}")
        positive_ids, negative_ids, _ = load_annotations(Path(annotations_path))

        logger.info(f"Loading detections: {detections_path}")
        detections = load_detections(Path(detections_path))

        # Extract features
        logger.info("\nExtracting features...")

        # Positive samples
        X_pos, valid_pos, feature_names = extract_feature_matrix(
            detections, list(positive_ids), exclude_prefixes=exclude_prefixes
        )
        y_pos = np.ones(len(valid_pos), dtype=np.int32)

        # Negative samples
        X_neg, valid_neg, _ = extract_feature_matrix(
            detections, list(negative_ids), feature_names=feature_names,
            exclude_prefixes=exclude_prefixes
        )
        y_neg = np.zeros(len(valid_neg), dtype=np.int32)

        logger.info(f"  Matched positive: {len(valid_pos)}/{len(positive_ids)}")
        logger.info(f"  Matched negative: {len(valid_neg)}/{len(negative_ids)}")

        if len(valid_pos) == 0 or len(valid_neg) == 0:
            raise ValueError("No matching samples found! Check UIDs match.")

        # Combine
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])
        uids = valid_pos + valid_neg

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"\nDataset shape: {X.shape}")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Positive samples: {(y == 1).sum()}")
    logger.info(f"  Negative samples: {(y == 0).sum()}")

    # Feature breakdown
    n_morph = len([f for f in feature_names
                   if not f.startswith('sam2_') and not f.startswith('resnet_')])
    n_sam2 = len([f for f in feature_names if f.startswith('sam2_')])
    n_resnet = len([f for f in feature_names if f.startswith('resnet_')])
    logger.info(f"  Morphological: {n_morph}")
    logger.info(f"  SAM2 embeddings: {n_sam2}")
    logger.info(f"  ResNet features: {n_resnet}")

    # Apply stratified sampling by size if requested
    if stratify_by_size:
        # Need detections to get diameters
        if sklearn_json_path:
            logger.warning("Cannot stratify by size from sklearn JSON - need detections file for diameters")
            logger.warning("Falling back to standard label stratification")
        elif detections_path:
            logger.info("\n" + "="*50)
            logger.info("STRATIFIED SAMPLING BY VESSEL SIZE")
            logger.info("="*50)

            detections = load_detections(Path(detections_path))
            diameters = extract_diameters(detections, uids)

            # Analyze original distribution
            analyze_size_distribution(diameters, y, prefix="Original ")

            # Apply stratified sampling
            X, y, uids, diameters = stratified_sample_by_size(
                X, y, uids, diameters,
                samples_per_class=samples_per_size_class,
                min_samples_per_class=10,
                random_seed=random_seed,
            )

            # Analyze balanced distribution
            analyze_size_distribution(diameters, y, prefix="Balanced ")

            logger.info(f"\nAfter stratification: {len(X)} samples")
            logger.info(f"  Positive: {(y == 1).sum()}")
            logger.info(f"  Negative: {(y == 0).sum()}")

    # Train/test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, uids_train, uids_test = train_test_split(
        X, y, uids, test_size=test_size, random_state=random_seed, stratify=y
    )

    logger.info(f"\nTrain/test split (seed={random_seed}, test_size={test_size}):")
    logger.info(f"  Train: {len(X_train)} samples ({(y_train==1).sum()} pos, {(y_train==0).sum()} neg)")
    logger.info(f"  Test: {len(X_test)} samples ({(y_test==1).sum()} pos, {(y_test==0).sum()} neg)")

    # Standardize features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save outputs
    logger.info(f"\nSaving to: {output_dir}")

    # 1. NumPy arrays (for direct loading)
    np.save(output_dir / 'X_train.npy', X_train_scaled)
    np.save(output_dir / 'X_test.npy', X_test_scaled)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_test.npy', y_test)
    logger.info("  Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")

    # 2. Raw (unscaled) arrays
    np.save(output_dir / 'X_train_raw.npy', X_train)
    np.save(output_dir / 'X_test_raw.npy', X_test)
    logger.info("  Saved: X_train_raw.npy, X_test_raw.npy (unscaled)")

    # 3. Pickle with scaler and metadata
    metadata = {
        'scaler': scaler,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_positive': int((y == 1).sum()),
        'n_negative': int((y == 0).sum()),
        'test_size': test_size,
        'random_seed': random_seed,
        'morph_only': morph_only,
        'uids_train': uids_train,
        'uids_test': uids_test,
    }

    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    logger.info("  Saved: metadata.pkl (scaler, feature_names, etc.)")

    # 4. CSV for inspection (first 50 features max)
    import csv
    csv_features = feature_names[:50] if len(feature_names) > 50 else feature_names
    csv_path = output_dir / 'training_data_preview.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['uid', 'label', 'split'] + csv_features)

        for uid, label, row in zip(uids_train, y_train, X_train):
            writer.writerow([uid, label, 'train'] + list(row[:len(csv_features)]))
        for uid, label, row in zip(uids_test, y_test, X_test):
            writer.writerow([uid, label, 'test'] + list(row[:len(csv_features)]))

    logger.info(f"  Saved: training_data_preview.csv (first {len(csv_features)} features)")

    # 5. Full training data JSON
    training_json = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'train': {
            'X': X_train.tolist(),
            'y': y_train.tolist(),
            'uids': uids_train,
        },
        'test': {
            'X': X_test.tolist(),
            'y': y_test.tolist(),
            'uids': uids_test,
        },
        'config': {
            'test_size': test_size,
            'random_seed': random_seed,
            'morph_only': morph_only,
        }
    }

    with open(output_dir / 'training_data.json', 'w') as f:
        json.dump(training_json, f)
    logger.info("  Saved: training_data.json")

    # 6. Feature names list
    with open(output_dir / 'feature_names.txt', 'w') as f:
        for fn in feature_names:
            f.write(fn + '\n')
    logger.info("  Saved: feature_names.txt")

    logger.info("\n" + "="*70)
    logger.info("PREPARATION COMPLETE")
    logger.info("="*70)

    logger.info("\nTo train a Random Forest classifier:")
    logger.info(f"""
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import pickle

    # Load data
    X_train = np.load('{output_dir}/X_train.npy')
    y_train = np.load('{output_dir}/y_train.npy')
    X_test = np.load('{output_dir}/X_test.npy')
    y_test = np.load('{output_dir}/y_test.npy')

    # Load metadata
    with open('{output_dir}/metadata.pkl', 'rb') as f:
        meta = pickle.load(f)

    # Train
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Evaluate
    accuracy = rf.score(X_test, y_test)
    print(f'Test accuracy: {{accuracy:.4f}}')
    """)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Random Forest training data from vessel annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From separate annotation and detection files:
    python prepare_rf_training_data.py \\
        --annotations vessel_annotations.json \\
        --detections vessel_detections.json \\
        --output-dir ./rf_data

    # From sklearn JSON exported from HTML interface:
    python prepare_rf_training_data.py \\
        --sklearn-json vessel_rf_sklearn.json \\
        --output-dir ./rf_data

    # Morphological features only (no embeddings):
    python prepare_rf_training_data.py \\
        --sklearn-json vessel_rf_sklearn.json \\
        --output-dir ./rf_data \\
        --morph-only
        """
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_argument_group('Input options')
    input_group.add_argument(
        '--annotations', '-a',
        help='Path to annotation JSON file (positive/negative/unsure lists)'
    )
    input_group.add_argument(
        '--detections', '-d',
        help='Path to detection JSON file with features'
    )
    input_group.add_argument(
        '--sklearn-json', '-s',
        help='Path to sklearn-format JSON (alternative to annotations+detections)'
    )

    # Output options
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument(
        '--output-dir', '-o',
        default='./rf_training_data',
        help='Output directory (default: ./rf_training_data)'
    )

    # Feature options
    feature_group = parser.add_argument_group('Feature options')
    feature_group.add_argument(
        '--morph-only',
        action='store_true',
        help='Use only morphological features (exclude SAM2/ResNet embeddings)'
    )
    feature_group.add_argument(
        '--no-sam2',
        action='store_true',
        help='Exclude SAM2 embeddings'
    )
    feature_group.add_argument(
        '--no-resnet',
        action='store_true',
        help='Exclude ResNet features'
    )

    # Split options
    split_group = parser.add_argument_group('Train/test split options')
    split_group.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for test set (default: 0.2)'
    )
    split_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    # Stratification options
    stratify_group = parser.add_argument_group('Stratification options')
    stratify_group.add_argument(
        '--stratify-by-size',
        action='store_true',
        help='Balance training data across vessel size classes (capillary/arteriole/small_artery/artery)'
    )
    stratify_group.add_argument(
        '--samples-per-size-class',
        type=int,
        default=None,
        help='Target samples per size class (default: auto = min class count)'
    )

    args = parser.parse_args()

    # Validate inputs
    if args.sklearn_json:
        if args.annotations or args.detections:
            logger.warning("sklearn-json provided, ignoring annotations/detections")
    else:
        if not args.annotations or not args.detections:
            parser.error("Must provide either --sklearn-json OR both "
                        "--annotations and --detections")

    prepare_training_data(
        annotations_path=args.annotations,
        detections_path=args.detections,
        sklearn_json_path=args.sklearn_json,
        output_dir=args.output_dir,
        morph_only=args.morph_only,
        include_sam2=not args.no_sam2,
        include_resnet=not args.no_resnet,
        test_size=args.test_size,
        random_seed=args.seed,
        stratify_by_size=args.stratify_by_size,
        samples_per_size_class=args.samples_per_size_class,
    )


if __name__ == '__main__':
    main()
