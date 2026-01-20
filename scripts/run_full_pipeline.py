#!/usr/bin/env python3
"""
Full vessel detection and classification pipeline.

This script implements a 3-stage vessel detection pipeline:

Stage 1: Candidate Detection (Permissive Mode)
    - Run vessel detection with low thresholds to capture all potential vessels
    - Accepts more false positives to minimize false negatives
    - Output: vessel candidates with features

Stage 2: Vessel Detection Classifier
    - Apply trained VesselDetectorRF to filter candidates
    - Binary classification: vessel vs non-vessel
    - Removes false positives (artifacts, noise, other structures)
    - Output: confirmed vessels only

Stage 3: Vessel Type Classification
    - Apply ArteryVeinClassifier to confirmed vessels
    - Classifies vessels as artery or vein
    - Uses wall thickness, diameter ratios, circularity
    - Output: final results with vessel type labels

Usage:
    # Run full pipeline on a detection results file
    python scripts/run_full_pipeline.py \\
        --input /path/to/candidate_detections.json \\
        --vessel-detector /path/to/vessel_detector.joblib \\
        --artery-vein /path/to/artery_vein_classifier.joblib \\
        --output /path/to/final_results.json

    # With confidence thresholds
    python scripts/run_full_pipeline.py \\
        --input candidate_detections.json \\
        --vessel-detector vessel_detector.joblib \\
        --vessel-threshold 0.5 \\
        --output final_results.json

    # Run candidate detection from scratch (on image/tiles)
    python scripts/run_full_pipeline.py \\
        --image /path/to/image.czi \\
        --vessel-detector vessel_detector.joblib \\
        --artery-vein artery_vein_classifier.joblib \\
        --output final_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.classification.vessel_detector_rf import VesselDetectorRF
from segmentation.classification.artery_vein_classifier import ArteryVeinClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class VesselPipeline:
    """
    Three-stage vessel detection and classification pipeline.

    Stage 1: Candidate detection (run externally or via this pipeline)
    Stage 2: Vessel detection (filter false positives)
    Stage 3: Artery/vein classification (type classification)
    """

    def __init__(
        self,
        vessel_detector: Optional[VesselDetectorRF] = None,
        artery_vein_classifier: Optional[ArteryVeinClassifier] = None,
        vessel_threshold: float = 0.5,
        artery_vein_threshold: float = 0.5,
    ):
        """
        Initialize the pipeline.

        Args:
            vessel_detector: Trained VesselDetectorRF model (stage 2)
            artery_vein_classifier: Trained ArteryVeinClassifier model (stage 3)
            vessel_threshold: Confidence threshold for vessel detection
            artery_vein_threshold: Confidence threshold for artery/vein classification
        """
        self.vessel_detector = vessel_detector
        self.artery_vein_classifier = artery_vein_classifier
        self.vessel_threshold = vessel_threshold
        self.artery_vein_threshold = artery_vein_threshold

    @classmethod
    def load_from_files(
        cls,
        vessel_detector_path: Optional[str] = None,
        artery_vein_path: Optional[str] = None,
        vessel_threshold: float = 0.5,
        artery_vein_threshold: float = 0.5,
    ) -> 'VesselPipeline':
        """
        Load pipeline from saved model files.

        Args:
            vessel_detector_path: Path to vessel detector model
            artery_vein_path: Path to artery/vein classifier model
            vessel_threshold: Confidence threshold for vessel detection
            artery_vein_threshold: Confidence threshold for artery/vein

        Returns:
            Initialized VesselPipeline
        """
        vessel_detector = None
        artery_vein_classifier = None

        if vessel_detector_path:
            logger.info(f"Loading vessel detector from: {vessel_detector_path}")
            vessel_detector = VesselDetectorRF.load(vessel_detector_path)

        if artery_vein_path:
            logger.info(f"Loading artery/vein classifier from: {artery_vein_path}")
            artery_vein_classifier = ArteryVeinClassifier.load(artery_vein_path)

        return cls(
            vessel_detector=vessel_detector,
            artery_vein_classifier=artery_vein_classifier,
            vessel_threshold=vessel_threshold,
            artery_vein_threshold=artery_vein_threshold,
        )

    def run_stage2_vessel_detection(
        self,
        candidates: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Stage 2: Apply vessel detector to filter candidates.

        Args:
            candidates: List of candidate detection dicts with 'features'
            verbose: Print progress information

        Returns:
            Tuple of (confirmed_vessels, rejected_candidates)
        """
        if self.vessel_detector is None:
            logger.warning("No vessel detector loaded, skipping stage 2")
            return candidates, []

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info("STAGE 2: Vessel Detection (Filtering False Positives)")
            logger.info(f"{'='*60}")
            logger.info(f"Input candidates: {len(candidates)}")
            logger.info(f"Confidence threshold: {self.vessel_threshold}")

        confirmed = []
        rejected = []

        for cand in candidates:
            features = cand.get('features', cand)

            is_vessel, confidence = self.vessel_detector.predict_vessel(features)

            # Add detection results to candidate
            cand_result = cand.copy()
            cand_result['vessel_detection'] = {
                'is_vessel': is_vessel,
                'confidence': confidence,
                'threshold': self.vessel_threshold,
                'passed': is_vessel and confidence >= self.vessel_threshold,
            }

            if is_vessel and confidence >= self.vessel_threshold:
                confirmed.append(cand_result)
            else:
                rejected.append(cand_result)

        if verbose:
            logger.info(f"Confirmed vessels: {len(confirmed)}")
            logger.info(f"Rejected candidates: {len(rejected)}")

            if confirmed:
                confidences = [c['vessel_detection']['confidence'] for c in confirmed]
                logger.info(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")

        return confirmed, rejected

    def run_stage3_artery_vein(
        self,
        vessels: List[Dict[str, Any]],
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Stage 3: Classify confirmed vessels as artery or vein.

        Args:
            vessels: List of confirmed vessel dicts
            verbose: Print progress information

        Returns:
            List of vessels with artery/vein classification added
        """
        if self.artery_vein_classifier is None:
            logger.warning("No artery/vein classifier loaded, using rule-based fallback")
            return self._apply_rule_based_classification(vessels, verbose)

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info("STAGE 3: Artery/Vein Classification")
            logger.info(f"{'='*60}")
            logger.info(f"Input vessels: {len(vessels)}")

        results = []
        artery_count = 0
        vein_count = 0

        for vessel in vessels:
            features = vessel.get('features', vessel)

            vessel_type, confidence = self.artery_vein_classifier.predict(features)
            proba = self.artery_vein_classifier.predict_proba(features)

            # Add classification results
            result = vessel.copy()
            result['artery_vein'] = {
                'classification': vessel_type,
                'confidence': confidence,
                'probabilities': proba,
                'threshold': self.artery_vein_threshold,
            }

            # Also add to top-level for convenience
            result['vessel_type'] = vessel_type
            result['type_confidence'] = confidence

            results.append(result)

            if vessel_type == 'artery':
                artery_count += 1
            else:
                vein_count += 1

        if verbose:
            logger.info(f"Arteries: {artery_count}")
            logger.info(f"Veins: {vein_count}")

        return results

    def _apply_rule_based_classification(
        self,
        vessels: List[Dict[str, Any]],
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Apply rule-based artery/vein classification when no model is available.
        """
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info("STAGE 3: Rule-Based Artery/Vein Classification")
            logger.info(f"{'='*60}")

        results = []
        artery_count = 0
        vein_count = 0

        for vessel in vessels:
            features = vessel.get('features', vessel)

            vessel_type, confidence = ArteryVeinClassifier.rule_based_classify(features)

            result = vessel.copy()
            result['artery_vein'] = {
                'classification': vessel_type,
                'confidence': confidence,
                'method': 'rule_based',
            }
            result['vessel_type'] = vessel_type
            result['type_confidence'] = confidence

            results.append(result)

            if vessel_type == 'artery':
                artery_count += 1
            else:
                vein_count += 1

        if verbose:
            logger.info(f"Arteries: {artery_count}")
            logger.info(f"Veins: {vein_count}")

        return results

    def run_full_pipeline(
        self,
        candidates: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete 3-stage pipeline.

        Args:
            candidates: List of stage 1 candidate detections
            verbose: Print progress information

        Returns:
            Dictionary with:
            - 'vessels': List of confirmed and classified vessels
            - 'rejected': List of rejected candidates
            - 'summary': Pipeline summary statistics
        """
        start_time = datetime.now()

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info("FULL VESSEL DETECTION PIPELINE")
            logger.info(f"{'='*60}")
            logger.info(f"Stage 1 candidates: {len(candidates)}")

        # Stage 2: Vessel detection
        confirmed, rejected = self.run_stage2_vessel_detection(candidates, verbose)

        # Stage 3: Artery/vein classification
        classified = self.run_stage3_artery_vein(confirmed, verbose)

        # Calculate summary
        elapsed = (datetime.now() - start_time).total_seconds()

        summary = {
            'input_candidates': len(candidates),
            'stage2_confirmed': len(confirmed),
            'stage2_rejected': len(rejected),
            'stage2_precision_estimate': len(confirmed) / max(len(candidates), 1),
            'arteries': sum(1 for v in classified if v.get('vessel_type') == 'artery'),
            'veins': sum(1 for v in classified if v.get('vessel_type') == 'vein'),
            'processing_time_seconds': elapsed,
            'vessel_threshold': self.vessel_threshold,
            'artery_vein_threshold': self.artery_vein_threshold,
        }

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info("PIPELINE SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Input candidates: {summary['input_candidates']}")
            logger.info(f"Confirmed vessels: {summary['stage2_confirmed']}")
            logger.info(f"Rejected: {summary['stage2_rejected']}")
            logger.info(f"Arteries: {summary['arteries']}")
            logger.info(f"Veins: {summary['veins']}")
            logger.info(f"Processing time: {elapsed:.2f}s")

        return {
            'vessels': classified,
            'rejected': rejected,
            'summary': summary,
        }


def load_candidates(input_path: Path) -> List[Dict[str, Any]]:
    """Load candidate detections from JSON file."""
    with open(input_path) as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        return data
    elif 'detections' in data:
        return data['detections']
    elif 'candidates' in data:
        return data['candidates']
    elif 'vessels' in data:
        return data['vessels']
    else:
        raise ValueError(f"Unknown format in {input_path}")


def save_results(
    results: Dict[str, Any],
    output_path: Path,
    include_rejected: bool = False
) -> None:
    """Save pipeline results to JSON file."""
    output_data = {
        'vessels': results['vessels'],
        'summary': results['summary'],
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'pipeline_version': '1.0',
        },
    }

    if include_rejected:
        output_data['rejected'] = results['rejected']

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def generate_summary_report(
    results: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate a human-readable summary report."""
    summary = results['summary']
    vessels = results['vessels']

    report_lines = [
        "=" * 60,
        "VESSEL DETECTION PIPELINE REPORT",
        "=" * 60,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DETECTION SUMMARY",
        "-" * 40,
        f"  Input candidates:    {summary['input_candidates']}",
        f"  Confirmed vessels:   {summary['stage2_confirmed']}",
        f"  Rejected:            {summary['stage2_rejected']}",
        f"  False positive rate: {(1 - summary['stage2_precision_estimate'])*100:.1f}%",
        "",
        "CLASSIFICATION SUMMARY",
        "-" * 40,
        f"  Arteries:            {summary['arteries']}",
        f"  Veins:               {summary['veins']}",
        "",
    ]

    # Add size statistics if available
    if vessels:
        diameters = [v.get('features', v).get('outer_diameter_um', 0) for v in vessels]
        diameters = [d for d in diameters if d > 0]

        if diameters:
            report_lines.extend([
                "SIZE STATISTICS",
                "-" * 40,
                f"  Mean diameter:       {np.mean(diameters):.1f} um",
                f"  Median diameter:     {np.median(diameters):.1f} um",
                f"  Min diameter:        {np.min(diameters):.1f} um",
                f"  Max diameter:        {np.max(diameters):.1f} um",
                "",
            ])

    # Confidence statistics
    vessel_confidences = [
        v.get('vessel_detection', {}).get('confidence', 0)
        for v in vessels
    ]
    type_confidences = [v.get('type_confidence', 0) for v in vessels]

    if vessel_confidences:
        report_lines.extend([
            "CONFIDENCE STATISTICS",
            "-" * 40,
            f"  Vessel detection (mean):  {np.mean(vessel_confidences):.3f}",
            f"  Vessel detection (min):   {np.min(vessel_confidences):.3f}",
            f"  Type classification:      {np.mean(type_confidences):.3f}",
            "",
        ])

    report_lines.extend([
        "PROCESSING",
        "-" * 40,
        f"  Time:                {summary['processing_time_seconds']:.2f} seconds",
        f"  Vessel threshold:    {summary['vessel_threshold']}",
        "",
        "=" * 60,
    ])

    report = "\n".join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Summary report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run full vessel detection and classification pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with pre-detected candidates
    python run_full_pipeline.py \\
        --input candidates.json \\
        --vessel-detector vessel_detector.joblib \\
        --artery-vein artery_vein_classifier.joblib \\
        --output results.json

    # With custom thresholds
    python run_full_pipeline.py \\
        --input candidates.json \\
        --vessel-detector vessel_detector.joblib \\
        --vessel-threshold 0.6 \\
        --output results.json

    # Output only confirmed vessels (exclude rejected)
    python run_full_pipeline.py \\
        --input candidates.json \\
        --vessel-detector vessel_detector.joblib \\
        --output results.json
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input JSON with candidate detections'
    )
    parser.add_argument(
        '--vessel-detector', '-v',
        help='Path to trained vessel detector model (.joblib)'
    )
    parser.add_argument(
        '--artery-vein', '-a',
        help='Path to trained artery/vein classifier (.joblib)'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--vessel-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for vessel detection (default: 0.5)'
    )
    parser.add_argument(
        '--artery-vein-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for artery/vein classification (default: 0.5)'
    )
    parser.add_argument(
        '--include-rejected',
        action='store_true',
        help='Include rejected candidates in output'
    )
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate human-readable summary report'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    # Load pipeline
    pipeline = VesselPipeline.load_from_files(
        vessel_detector_path=args.vessel_detector,
        artery_vein_path=args.artery_vein,
        vessel_threshold=args.vessel_threshold,
        artery_vein_threshold=args.artery_vein_threshold,
    )

    # Load candidates
    logger.info(f"\nLoading candidates from: {args.input}")
    candidates = load_candidates(Path(args.input))
    logger.info(f"Loaded {len(candidates)} candidates")

    # Run pipeline
    results = pipeline.run_full_pipeline(candidates, verbose=not args.quiet)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_results(results, output_path, include_rejected=args.include_rejected)

    # Generate report if requested
    if args.generate_report:
        report_path = output_path.with_suffix('.report.txt')
        generate_summary_report(results, report_path)

    logger.info("\nPipeline complete!")


if __name__ == '__main__':
    main()
