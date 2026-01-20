"""
Vessel classification module.

Provides ML-based vessel classification with a multi-stage pipeline:

Stage 1 - Vessel Detection (vessel_detector_rf.py):
    Binary classification: Is this candidate a vessel or a false positive?
    Uses VesselDetectorRF with morphological and intensity features.

Stage 2 - Vessel Type Classification (multiple options):
    a) VesselClassifier (vessel_classifier.py):
       Classifies vessels as capillary, arteriole, or artery (3 classes).
    b) ArteryVeinClassifier (artery_vein_classifier.py):
       Binary classification: artery vs vein (2 classes).
    c) VesselTypeClassifier (vessel_type_classifier.py):
       Full 6-class classifier: artery, arteriole, vein, capillary,
       lymphatic, collecting_lymphatic. Uses marker profiles (SMA, CD31, LYVE1).

Usage:
    # Stage 1: Vessel Detection
    from segmentation.classification import VesselDetectorRF

    detector = VesselDetectorRF()
    detector.train_from_files('annotations.json', 'detections.json')
    is_vessel, confidence = detector.predict_vessel(features)

    # Stage 2a: Vessel Type (capillary/arteriole/artery)
    from segmentation.classification import VesselClassifier

    classifier = VesselClassifier()
    classifier.train(X, y, feature_names)
    predictions, confidence = classifier.predict(features)

    # Stage 2b: Artery vs Vein (binary)
    from segmentation.classification import ArteryVeinClassifier

    av_classifier = ArteryVeinClassifier()
    av_classifier.train(X, y)
    vessel_type, confidence = av_classifier.predict(features)

    # Stage 2c: Full 6-class Vessel Type (with marker profiles)
    from segmentation.classification import VesselTypeClassifier

    type_classifier = VesselTypeClassifier()
    type_classifier.train(X, y, feature_names)
    vessel_type, confidence = type_classifier.predict(features)
    # Or get top-k predictions:
    top_types = type_classifier.predict_top_k(features, k=3)
"""

from .vessel_classifier import (
    VesselClassifier,
    VESSEL_CORE_FEATURES,
    MORPHOLOGICAL_FEATURES,
    DEFAULT_FEATURES,
    SAM2_FEATURES,
    RESNET_FEATURES,
    FULL_FEATURES,
)
from .vessel_detector_rf import (
    VesselDetectorRF,
    train_vessel_detector,
    predict_vessel,
    VESSEL_DETECTION_FEATURES,
    MINIMAL_DETECTION_FEATURES,
)
from .artery_vein_classifier import (
    ArteryVeinClassifier,
    ARTERY_VEIN_FEATURES,
    MINIMAL_ARTERY_VEIN_FEATURES,
)
from .vessel_type_classifier import (
    VesselTypeClassifier,
    TYPE_FEATURES,
    EXTENDED_TYPE_FEATURES,
    MINIMAL_TYPE_FEATURES,
    VESSEL_TYPES,
    VESSEL_TYPE_CHARACTERISTICS,
    classify_vessel_type,
    get_vessel_type_description,
)
from .feature_selection import (
    analyze_feature_importance,
    select_optimal_features,
    cross_validate_features,
)

__all__ = [
    # Vessel Detection (Stage 1)
    'VesselDetectorRF',
    'train_vessel_detector',
    'predict_vessel',
    'VESSEL_DETECTION_FEATURES',
    'MINIMAL_DETECTION_FEATURES',
    # Vessel Type Classification (Stage 2a) - 3 classes
    'VesselClassifier',
    'VESSEL_CORE_FEATURES',
    'MORPHOLOGICAL_FEATURES',
    'DEFAULT_FEATURES',
    'SAM2_FEATURES',
    'RESNET_FEATURES',
    'FULL_FEATURES',
    # Artery/Vein Classification (Stage 2b) - 2 classes
    'ArteryVeinClassifier',
    'ARTERY_VEIN_FEATURES',
    'MINIMAL_ARTERY_VEIN_FEATURES',
    # Full Vessel Type Classification (Stage 2c) - 6 classes with marker profiles
    'VesselTypeClassifier',
    'TYPE_FEATURES',
    'EXTENDED_TYPE_FEATURES',
    'MINIMAL_TYPE_FEATURES',
    'VESSEL_TYPES',
    'VESSEL_TYPE_CHARACTERISTICS',
    'classify_vessel_type',
    'get_vessel_type_description',
    # Feature Selection Utilities
    'analyze_feature_importance',
    'select_optimal_features',
    'cross_validate_features',
]
