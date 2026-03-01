"""Detection strategy creation, classifier loading, and parameter building.

Functions for creating cell-type-specific detection strategies, loading trained
classifiers (NMJ/islet/tissue_pattern RF, vessel binary+6-type), and building
detection parameter dicts from CLI arguments.
"""

from pathlib import Path

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def create_strategy_for_cell_type(cell_type, params, pixel_size_um):
    """Create the appropriate detection strategy for a cell type.

    Args:
        cell_type: One of 'nmj', 'mk', 'cell', 'vessel', 'mesothelium', 'islet', 'tissue_pattern'
        params: Cell-type specific parameters dict
        pixel_size_um: Pixel size in microns

    Returns:
        DetectionStrategy instance

    Raises:
        ValueError: If cell_type is not supported by the new strategy pattern
    """
    from segmentation.processing.strategy_factory import create_strategy
    return create_strategy(
        cell_type=cell_type,
        strategy_params=params,
        extract_deep_features=params.get('extract_deep_features', False),
        extract_sam2_embeddings=params.get('extract_sam2_embeddings', True),
        pixel_size_um=pixel_size_um,
    )


def apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier):
    """Apply vessel binary + 6-class classifiers to detection features in-place."""
    from segmentation.classification.vessel_classifier import VesselClassifier

    if vessel_classifier is not None:
        for feat in features_list:
            try:
                vessel_type, confidence = vessel_classifier.predict(feat['features'])
                feat['features']['vessel_type'] = vessel_type
                feat['features']['vessel_type_confidence'] = float(confidence)
                feat['features']['classification_method'] = 'ml'
            except Exception as e:
                vessel_type, confidence = VesselClassifier.rule_based_classify(feat['features'])
                feat['features']['vessel_type'] = vessel_type
                feat['features']['vessel_type_confidence'] = float(confidence)
                feat['features']['classification_method'] = 'rule_based_fallback'
                logger.debug(f"ML classification failed, using rule-based: {e}")

    if vessel_type_classifier is not None:
        for feat in features_list:
            try:
                vessel_type, confidence = vessel_type_classifier.predict(feat['features'])
                probs = vessel_type_classifier.predict_proba(feat['features'])
                feat['features']['vessel_type_6class'] = vessel_type
                feat['features']['vessel_type_6class_confidence'] = float(confidence)
                feat['features']['vessel_type_6class_probabilities'] = {
                    k: float(v) for k, v in probs.items()
                } if probs else {}
                feat['features']['classification_method_6class'] = 'ml_vessel_type_classifier'
            except Exception as e:
                try:
                    vessel_type, confidence = vessel_type_classifier.rule_based_classify(feat['features'])
                    feat['features']['vessel_type_6class'] = vessel_type
                    feat['features']['vessel_type_6class_confidence'] = float(confidence)
                    feat['features']['classification_method_6class'] = 'rule_based_fallback'
                except Exception as e2:
                    logger.debug(f"VesselTypeClassifier failed: {e}, {e2}")


def load_classifier_into_detector(args, detector):
    """Load NMJ/islet/tissue_pattern classifier into detector.models.

    Supports CNN (.pth) and RF (.pkl) classifiers. The loaded classifier is
    stored in detector.models for use during tile processing.

    Args:
        args: Parsed CLI args (reads cell_type, nmj_classifier, islet_classifier, tp_classifier)
        detector: CellDetector instance to populate

    Returns:
        classifier_loaded: bool indicating whether a classifier was successfully loaded
    """
    classifier_loaded = False

    if args.cell_type == 'nmj' and getattr(args, 'nmj_classifier', None):
        from segmentation.detection.strategies.nmj import load_classifier

        logger.info(f"Loading NMJ classifier from {args.nmj_classifier}...")
        classifier_data = load_classifier(args.nmj_classifier, device=detector.device)

        if classifier_data['type'] == 'cnn':
            # CNN classifier - use transform pipeline
            from torchvision import transforms
            classifier_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            detector.models['classifier'] = classifier_data['model']
            detector.models['classifier_type'] = 'cnn'
            detector.models['transform'] = classifier_transform
            detector.models['device'] = classifier_data['device']
            logger.info("CNN classifier loaded successfully")
            classifier_loaded = True
        else:
            # RF classifier - use features directly
            # New format uses 'pipeline', legacy uses 'model'
            if 'pipeline' in classifier_data:
                detector.models['classifier'] = classifier_data['pipeline']
                detector.models['scaler'] = None  # Pipeline handles scaling internally
            else:
                detector.models['classifier'] = classifier_data['model']
                detector.models['scaler'] = classifier_data.get('scaler')
            detector.models['classifier_type'] = 'rf'
            detector.models['feature_names'] = classifier_data['feature_names']
            logger.info(f"RF classifier loaded successfully ({len(classifier_data['feature_names'])} features)")
            classifier_loaded = True

    elif args.cell_type == 'islet' and getattr(args, 'islet_classifier', None):
        from segmentation.detection.strategies.nmj import load_nmj_rf_classifier
        logger.info(f"Loading islet RF classifier from {args.islet_classifier}...")
        classifier_data = load_nmj_rf_classifier(args.islet_classifier)
        # load_nmj_rf_classifier always returns 'pipeline' key (wraps legacy format)
        detector.models['classifier'] = classifier_data['pipeline']
        detector.models['scaler'] = None
        detector.models['classifier_type'] = 'rf'
        detector.models['feature_names'] = classifier_data['feature_names']
        logger.info(f"Islet RF classifier loaded ({len(classifier_data['feature_names'])} features)")
        classifier_loaded = True

    elif args.cell_type == 'tissue_pattern' and getattr(args, 'tp_classifier', None):
        from segmentation.detection.strategies.nmj import load_nmj_rf_classifier
        logger.info(f"Loading tissue_pattern RF classifier from {args.tp_classifier}...")
        classifier_data = load_nmj_rf_classifier(args.tp_classifier)
        detector.models['classifier'] = classifier_data['pipeline']
        detector.models['scaler'] = None
        detector.models['classifier_type'] = 'rf'
        detector.models['feature_names'] = classifier_data['feature_names']
        logger.info(f"Tissue pattern RF classifier loaded ({len(classifier_data['feature_names'])} features)")
        classifier_loaded = True

    return classifier_loaded


def build_detection_params(args, pixel_size_um):
    """Build cell-type-specific detection parameter dict from CLI args.

    Args:
        args: Parsed CLI args
        pixel_size_um: Pixel size from CZI metadata

    Returns:
        params: dict of detection parameters
    """
    if args.cell_type == 'nmj':
        params = {
            'intensity_percentile': args.intensity_percentile,
            'min_area': args.min_area,
            'min_skeleton_length': args.min_skeleton_length,
            'max_solidity': args.max_solidity,
            'extract_deep_features': getattr(args, 'extract_deep_features', False),
        }
    elif args.cell_type == 'mk':
        params = {
            'mk_min_area': args.mk_min_area,
            'mk_max_area': args.mk_max_area,
        }
    elif args.cell_type == 'cell':
        params = {
            'min_area_um': args.min_cell_area,
            'max_area_um': args.max_cell_area,
        }
        if args.cellpose_input_channels:
            try:
                parts = args.cellpose_input_channels.split(',')
                params['cellpose_input_channels'] = [int(parts[0]), int(parts[1])]
            except (ValueError, IndexError):
                raise ValueError(f"--cellpose-input-channels must be two integers like '1,0', got '{args.cellpose_input_channels}'")
    elif args.cell_type == 'vessel':
        params = {
            'min_vessel_diameter_um': args.min_vessel_diameter,
            'max_vessel_diameter_um': args.max_vessel_diameter,
            'min_wall_thickness_um': args.min_wall_thickness,
            'max_aspect_ratio': args.max_aspect_ratio,
            'min_circularity': args.min_circularity,
            'min_ring_completeness': args.min_ring_completeness,
            'pixel_size_um': pixel_size_um,
            'classify_vessel_types': args.classify_vessel_types,
            'use_ml_classification': args.use_ml_classification,
            'vessel_classifier_path': args.vessel_classifier_path,
            'candidate_mode': args.candidate_mode,
            'lumen_first': getattr(args, 'lumen_first', False),
            'ring_only': getattr(args, 'ring_only', False),
            'parallel_detection': getattr(args, 'parallel_detection', False),
            'parallel_workers': getattr(args, 'parallel_workers', 3),
            'multi_marker': getattr(args, 'multi_marker', False),
            'smooth_contours': not getattr(args, 'no_smooth_contours', False),
            'smooth_contours_factor': getattr(args, 'smooth_contours_factor', 3.0),
        }
    elif args.cell_type == 'mesothelium':
        params = {
            'target_chunk_area_um2': args.target_chunk_area,
            'min_ribbon_width_um': args.min_ribbon_width,
            'max_ribbon_width_um': args.max_ribbon_width,
            'min_fragment_area_um2': args.min_fragment_area,
            'pixel_size_um': pixel_size_um,
        }
    elif args.cell_type == 'islet':
        nuclei_only = getattr(args, 'nuclei_only', False)
        params = {
            'membrane_channel': None if nuclei_only else getattr(args, 'membrane_channel', 1),
            'nuclear_channel': getattr(args, 'nuclear_channel', 4),
            'extract_deep_features': getattr(args, 'extract_deep_features', False),
            'marker_signal_factor': getattr(args, 'marker_signal_factor', 2.0),
        }
    elif args.cell_type == 'tissue_pattern':
        params = {
            'detection_channels': [int(x) for x in args.tp_detection_channels.split(',')],
            'nuclear_channel': getattr(args, 'tp_nuclear_channel', 4),
            'min_area_um': getattr(args, 'tp_min_area', 20.0),
            'max_area_um': getattr(args, 'tp_max_area', 300.0),
            'extract_deep_features': getattr(args, 'extract_deep_features', False),
        }
    else:
        raise ValueError(f"Unknown cell type: {args.cell_type}")

    return params


def load_vessel_classifiers(args):
    """Load vessel binary and 6-type classifiers if configured.

    Args:
        args: Parsed CLI args (reads use_ml_classification, vessel_classifier_path,
              vessel_type_classifier, cell_type)

    Returns:
        (vessel_classifier, vessel_type_classifier) tuple, either may be None
    """
    from segmentation.classification.vessel_classifier import VesselClassifier
    from segmentation.classification.vessel_type_classifier import VesselTypeClassifier

    vessel_classifier = None
    vessel_type_classifier = None

    # Load vessel binary classifier if ML classification requested
    if args.cell_type == 'vessel' and args.use_ml_classification:
        classifier_path = args.vessel_classifier_path
        if classifier_path and Path(classifier_path).exists():
            try:
                vessel_classifier = VesselClassifier.load(classifier_path)
                logger.info(f"Loaded vessel classifier from: {classifier_path}")
                _cv_acc = vessel_classifier.metrics.get('cv_accuracy_mean', 'N/A')
                if isinstance(_cv_acc, (int, float)):
                    logger.info(f"  CV accuracy: {_cv_acc:.4f}")
                else:
                    logger.info(f"  CV accuracy: {_cv_acc}")
            except Exception as e:
                logger.warning(f"Failed to load vessel classifier: {e}")
                logger.warning("Falling back to rule-based classification")
        else:
            logger.warning("--use-ml-classification specified but no model path provided or file not found")
            logger.warning("Falling back to rule-based classification")
            if args.classify_vessel_types:
                logger.info("Using rule-based diameter thresholds for vessel classification")

    # Load VesselTypeClassifier if path provided (for multi-marker 6-type classification)
    if args.cell_type == 'vessel' and getattr(args, 'vessel_type_classifier', None):
        classifier_path = args.vessel_type_classifier
        if Path(classifier_path).exists():
            try:
                vessel_type_classifier = VesselTypeClassifier.load(classifier_path)
                logger.info(f"Loaded VesselTypeClassifier from: {classifier_path}")
                if vessel_type_classifier.metrics:
                    accuracy = vessel_type_classifier.metrics.get('cv_accuracy_mean', 'N/A')
                    if isinstance(accuracy, float):
                        logger.info(f"  CV accuracy: {accuracy:.4f}")
                    else:
                        logger.info(f"  CV accuracy: {accuracy}")
            except Exception as e:
                logger.warning(f"Failed to load VesselTypeClassifier: {e}")
                vessel_type_classifier = None
        else:
            logger.warning(f"VesselTypeClassifier path does not exist: {classifier_path}")

    return vessel_classifier, vessel_type_classifier
