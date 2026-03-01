"""Argument parser construction and postprocessing for the segmentation pipeline CLI.

Contains build_parser() which defines all CLI arguments, and postprocess_args()
which applies cell-type-dependent defaults, parses compound args, and validates.
"""

import argparse
from pathlib import Path

import torch


def build_parser():
    """Build the argument parser for the segmentation pipeline CLI.

    Returns:
        argparse.ArgumentParser with all arguments configured
    """
    parser = argparse.ArgumentParser(
        description='Unified Cell Segmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required (unless using utility commands like --stop-server or --server-status)
    parser.add_argument('--czi-path', type=str, required=False, help='Path to CZI file')
    parser.add_argument('--cell-type', type=str, default=None,
                        choices=['nmj', 'mk', 'cell', 'vessel', 'mesothelium', 'islet', 'tissue_pattern'],
                        help='Cell type to detect (not required if --show-metadata)')

    # CZI scene selection (multi-scene slides, e.g. brain with 2 tissue sections)
    parser.add_argument('--scene', type=int, default=0,
                        help='CZI scene index (0-based, default 0). '
                             'Multi-scene slides store separate tissue sections as scenes.')

    # Metadata inspection
    parser.add_argument('--show-metadata', action='store_true',
                        help='Show CZI channel/dimension info and exit (no processing)')

    # Performance options - RAM loading is the default for single slides (best for network mounts)
    parser.add_argument('--load-to-ram', action='store_true', default=True,
                        help='Load entire channel into RAM first (default: True for best performance on network mounts)')
    parser.add_argument('--no-ram', dest='load_to_ram', action='store_false',
                        help='[DEPRECATED - ignored] RAM loading is always used. This flag is kept for backward compatibility only.')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./output)')

    # Tile processing
    parser.add_argument('--tile-size', type=int, default=3000, help='Tile size in pixels')
    parser.add_argument('--tile-overlap', type=float, default=0.10, help='Tile overlap fraction (0.0-0.5, default: 0.10 = 10%% overlap)')
    parser.add_argument('--sample-fraction', type=float, default=1.0, help='Fraction of tissue tiles to process (default: 100%%)')
    parser.add_argument('--channel', type=int, default=None,
                        help='Primary channel index for detection (default: 1 for NMJ, 0 for MK/vessel/cell)')
    parser.add_argument('--all-channels', action='store_true',
                        help='Load all channels for multi-channel analysis (NMJ specificity checking)')
    parser.add_argument('--channel-names', type=str, default=None,
                        help='Comma-separated channel names for feature naming (e.g., "nuclear,sma,pm,cd31" or "nuclear,sma,pm,lyve1")')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose/debug logging')
    parser.add_argument('--photobleaching-correction', action='store_true',
                        help='Apply slide-wide photobleaching correction (fixes horizontal/vertical banding)')
    parser.add_argument('--norm-params-file', type=str, default=None,
                        help='Path to pre-computed Reinhard normalization params JSON (from compute_normalization_params.py). '
                             'Applies whole-slide Lab-space normalization before tile processing.')
    parser.add_argument('--normalize-features', action='store_true', default=True,
                        help='Apply flat-field illumination correction (default: ON)')
    parser.add_argument('--no-normalize-features', dest='normalize_features', action='store_false',
                        help='Disable flat-field correction (use raw intensities)')
    parser.add_argument('--html-normalization', choices=['tile', 'crop'], default='tile',
                        help='HTML crop normalization scope: tile=shared percentiles per tile, crop=per-crop (default: tile)')

    # NMJ parameters
    parser.add_argument('--intensity-percentile', type=float, default=98)
    parser.add_argument('--min-area', type=int, default=150)
    parser.add_argument('--min-skeleton-length', type=int, default=30)
    parser.add_argument('--max-solidity', type=float, default=0.85,
                        help='Maximum solidity for NMJ detection (branched structures have low solidity)')
    parser.add_argument('--nmj-classifier', type=str, default=None,
                        help='Path to trained NMJ classifier (.pth file)')
    parser.add_argument('--html-score-threshold', type=float, default=0.5,
                        help='Minimum rf_prediction score to show in HTML (default 0.5). '
                             'All detections still saved to JSON regardless. '
                             'Auto-set to 0.0 when no classifier is loaded (annotation run). '
                             'Use --html-score-threshold 0.0 to show ALL candidates explicitly.')
    parser.add_argument('--prior-annotations', type=str, default=None,
                        help='Path to prior annotations JSON file (from round-1 annotation). '
                             'Pre-loads annotations into HTML localStorage so round-1 labels '
                             'are visible during round-2 review after classifier training.')

    # MK parameters (area in um^2)
    parser.add_argument('--mk-min-area', type=float, default=200.0,
                        help='Minimum MK area in um^2 (default 200)')
    parser.add_argument('--mk-max-area', type=float, default=2000.0,
                        help='Maximum MK area in um^2 (default 2000)')

    # Cell strategy parameters
    parser.add_argument('--cellpose-input-channels', type=str, default=None,
                        help='Two CZI channel indices for 2-channel Cellpose: CYTO,NUC (e.g., 1,0). '
                             'Cyto = plasma membrane/cytoplasmic marker, Nuc = nuclear stain.')
    parser.add_argument('--min-cell-area', type=float, default=50.0,
                        help='Minimum cell area in um^2 for --cell-type cell (default 50)')
    parser.add_argument('--max-cell-area', type=float, default=200.0,
                        help='Maximum cell area in um^2 for --cell-type cell (default 200)')

    # Vessel parameters
    parser.add_argument('--min-vessel-diameter', type=float, default=10,
                        help='Minimum vessel outer diameter in um')
    parser.add_argument('--max-vessel-diameter', type=float, default=1000,
                        help='Maximum vessel outer diameter in um')
    parser.add_argument('--min-wall-thickness', type=float, default=2,
                        help='Minimum vessel wall thickness in um')
    parser.add_argument('--max-aspect-ratio', type=float, default=4.0,
                        help='Maximum aspect ratio (exclude longitudinal sections)')
    parser.add_argument('--min-circularity', type=float, default=0.3,
                        help='Minimum circularity for vessel detection')
    parser.add_argument('--min-ring-completeness', type=float, default=0.5,
                        help='Minimum ring completeness (fraction of SMA+ perimeter)')
    parser.add_argument('--cd31-channel', type=int, default=None,
                        help='CD31 channel index for vessel validation (optional)')
    parser.add_argument('--classify-vessel-types', action='store_true',
                        help='Auto-classify vessels by size (capillary/arteriole/artery) using rule-based method')
    parser.add_argument('--use-ml-classification', action='store_true',
                        help='Use ML-based vessel classification (requires trained model)')
    parser.add_argument('--vessel-classifier-path', type=str, default=None,
                        help='Path to trained vessel classifier (.joblib). If not provided with '
                             '--use-ml-classification, falls back to rule-based classification.')
    parser.add_argument('--candidate-mode', action='store_true',
                        help='Enable candidate generation mode for vessel detection. '
                             'Relaxes all thresholds to catch more potential vessels (higher recall). '
                             'Includes detection_confidence score (0-1) for each candidate. '
                             'Use for generating training data for manual annotation + RF classifier.')
    parser.add_argument('--lumen-first', action='store_true',
                        help='[DEPRECATED] Lumen-first detection now runs automatically as a '
                             'supplementary pass alongside ring detection. This flag is a no-op. '
                             'Use --ring-only to disable the supplementary lumen-first pass.')
    parser.add_argument('--ring-only', action='store_true',
                        help='Disable the supplementary lumen-first detection pass. '
                             'Only use Canny edge + contour hierarchy ring detection. '
                             'Useful if you know there are no great vessels in the tissue.')
    parser.add_argument('--parallel-detection', action='store_true',
                        help='Enable parallel multi-marker vessel detection. '
                             'Runs SMA, CD31, and LYVE1 detection in parallel using CPU threads. '
                             'Requires --channel-names to specify marker channels. '
                             'Example: --channel-names "nuclear,sma,cd31,lyve1" --parallel-detection')
    parser.add_argument('--parallel-workers', type=int, default=3,
                        help='Number of parallel workers for multi-marker detection (default: 3). '
                             'One worker per marker type (SMA, CD31, LYVE1).')
    parser.add_argument('--multi-marker', action='store_true',
                        help='Enable full multi-marker vessel detection pipeline. '
                             'Automatically enables --all-channels and --parallel-detection. '
                             'Detects SMA+ rings, CD31+ capillaries, and LYVE1+ lymphatics. '
                             'Merges overlapping candidates from different markers. '
                             'Extracts multi-channel features for downstream classification. '
                             'Example: --multi-marker --channel-names "nuclear,sma,cd31,lyve1"')
    parser.add_argument('--no-smooth-contours', action='store_true',
                        help='Disable B-spline contour smoothing (on by default). '
                             'Smoothing removes stair-step artifacts from coarse-scale detection.')
    parser.add_argument('--smooth-contours-factor', type=float, default=3.0,
                        help='Spline smoothing factor for vessel contours (default: 3.0). '
                             'Higher = smoother. 0 = interpolating spline (passes through all points).')
    parser.add_argument('--vessel-type-classifier', type=str, default=None,
                        help='Path to trained VesselTypeClassifier model (.joblib) for 6-type '
                             'vessel classification (artery/arteriole/vein/capillary/lymphatic/'
                             'collecting_lymphatic). Used with --multi-marker for automated '
                             'vessel type prediction based on marker profiles and morphology.')

    # Multi-scale vessel detection
    parser.add_argument('--multi-scale', action='store_true',
                        help='Enable multi-scale vessel detection. Detects at multiple resolutions '
                             '(1/8x, 1/4x, 1x) to capture all vessel sizes and avoid cross-tile '
                             'fragmentation. Large vessels are detected at coarse scale (1/8x) '
                             'where they fit within a single tile. Requires --cell-type vessel.')
    parser.add_argument('--scales', type=str, default='32,16,8,4,2',
                        help='Comma-separated scale factors for multi-scale detection (default: "32,16,8,4,2"). '
                             'Numbers represent downsampling factors: 32=1/32x (large arteries), '
                             '16=1/16x, 8=1/8x, 4=1/4x (medium), 2=1/2x (small vessels). '
                             'Detection runs coarse-to-fine with IoU deduplication.')
    parser.add_argument('--multiscale-iou-threshold', type=float, default=0.3,
                        help='IoU threshold for deduplicating vessels detected at multiple scales '
                             '(default: 0.3). If a vessel is detected at both coarse and fine scales '
                             'with IoU > threshold, the detection with larger contour area is kept.')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume multiscale run from checkpoints in a previous run directory. '
                             'Skips already-completed scales and reuses the output directory.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume pipeline from an existing run directory. '
                             'Auto-detects completed stages (tiles, detections, HTML) and skips them. '
                             'Requires --czi-path (for HTML crop rendering unless everything is done).')
    parser.add_argument('--force-html', action='store_true', default=False,
                        help='Force HTML regeneration even if html/ exists (use with --resume)')
    parser.add_argument('--force-detect', action='store_true', default=False,
                        help='Force re-detection even if tiles/ has data (use with --resume)')

    # Mesothelium parameters (for LMD chunking)
    parser.add_argument('--target-chunk-area', type=float, default=1500,
                        help='Target area for mesothelium chunks in um^2')
    parser.add_argument('--min-ribbon-width', type=float, default=5,
                        help='Minimum expected ribbon width in um')
    parser.add_argument('--max-ribbon-width', type=float, default=30,
                        help='Maximum expected ribbon width in um')
    parser.add_argument('--min-fragment-area', type=float, default=1500,
                        help='Skip mesothelium fragments smaller than this (um^2)')
    parser.add_argument('--add-fiducials', action='store_true', default=True,
                        help='Add calibration cross markers to LMD export')
    parser.add_argument('--no-fiducials', dest='add_fiducials', action='store_false',
                        help='Do not add calibration markers')

    # Islet parameters
    parser.add_argument('--membrane-channel', type=int, default=1,
                        help='Membrane marker channel index for islet Cellpose input (default: 1, AF633)')
    parser.add_argument('--nuclear-channel', type=int, default=4,
                        help='Nuclear marker channel index for islet Cellpose input (default: 4, DAPI)')
    parser.add_argument('--islet-classifier', type=str, default=None,
                        help='Path to trained islet RF classifier (.pkl)')
    parser.add_argument('--islet-display-channels', type=str, default='2,3,5',
                        help='Comma-separated R,G,B channel indices for islet HTML display (default: 2,3,5). '
                             'Channels are mapped to R/G/B in order.')
    parser.add_argument('--islet-marker-channels', type=str, default='gcg:2,ins:3,sst:5',
                        help='Marker-to-channel mapping for islet classification, as name:index pairs. '
                             'Format: "gcg:2,ins:3,sst:5". Names are used in logs and legends.')
    parser.add_argument('--nuclei-only', action='store_true', default=False,
                        help='Nuclei-only mode for islet: use DAPI grayscale for Cellpose '
                             '(channels=[0,0]) instead of membrane+nuclear. SAM2 still runs.')
    parser.add_argument('--marker-signal-factor', type=float, default=2.0,
                        help='Pre-filter divisor for GMM threshold. Cells need marker '
                             'signal > auto_threshold/N to get full features + SAM2. '
                             'Higher = more permissive. 0 = disable. (default 2.0)')
    parser.add_argument('--marker-top-pct', type=float, default=5,
                        help='For percentile-method channels (see --marker-pct-channels), '
                             'classify the top N%% of cells as marker-positive. '
                             '(default 5 = 95th percentile)')
    parser.add_argument('--marker-pct-channels', type=str, default='sst',
                        help='Comma-separated marker names that use percentile-based '
                             'thresholding instead of GMM (default: sst)')
    parser.add_argument('--gmm-p-cutoff', type=float, default=0.75,
                        help='GMM posterior probability cutoff for marker classification. '
                             'Higher = stricter (fewer false positives). (default 0.75)')
    parser.add_argument('--ratio-min', type=float, default=1.5,
                        help='Dominant marker must be >= ratio_min * runner-up for '
                             'single-marker classification. Below -> "multi". (default 1.5)')
    parser.add_argument('--dedup-by-confidence', action='store_true', default=False,
                        help='Sort by confidence (score) instead of area during deduplication. '
                             'Default: sort by area (largest mask wins overlap).')

    # Tissue pattern parameters
    parser.add_argument('--tp-detection-channels', type=str, default='0,3',
                        help='Comma-separated channel indices to sum for tissue_pattern detection (default: 0,3 = Slc17a7+Gad1)')
    parser.add_argument('--tp-nuclear-channel', type=int, default=4,
                        help='Nuclear channel for tissue detection (default: 4, Hoechst)')
    parser.add_argument('--tp-display-channels', type=str, default='0,3,1',
                        help='Comma-separated R,G,B channel indices for HTML display (default: 0,3,1 = Slc17a7/Gad1/Htr2a)')
    parser.add_argument('--tp-classifier', type=str, default=None,
                        help='Path to trained tissue_pattern RF classifier (.pkl)')
    parser.add_argument('--tp-min-area', type=float, default=20.0,
                        help='Minimum cell area in um^2 for tissue_pattern (default 20)')
    parser.add_argument('--tp-max-area', type=float, default=300.0,
                        help='Maximum cell area in um^2 for tissue_pattern (default 300)')

    # Tissue detection
    parser.add_argument('--variance-threshold', type=float, default=None,
                        help='Manual variance threshold for tissue detection, overriding K-means calibration. '
                             'Use when auto-calibration is too strict (e.g. out-of-focus scenes). '
                             'Check logs for calibrated values to guide manual setting.')

    # Channel selection
    parser.add_argument('--channels', type=str, default=None,
                        help='Comma-separated list of CZI channel indices to load (e.g. "8,9,10,11"). '
                             'If not specified, all channels are loaded when --all-channels is active. '
                             'Use with multi-channel CZIs that have EDF/processing layers to avoid loading unnecessary data.')

    # Feature extraction options
    parser.add_argument('--extract-deep-features', action='store_true',
                        help='Extract ResNet and DINOv2 features (opt-in, default morph+SAM2 only)')
    parser.add_argument('--skip-deep-features', action='store_true',
                        help='Deprecated: deep features are off by default now. Use --extract-deep-features to enable.')

    # GPU processing (always uses multi-GPU infrastructure, even with 1 GPU)
    parser.add_argument('--multi-gpu', action='store_true', default=True,
                        help='[DEPRECATED - always enabled] Multi-GPU processing is now the only code path. '
                             'Use --num-gpus to control how many GPUs are used (default: auto-detect).')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: auto-detect via torch.cuda.device_count(), '
                             'minimum 1). The pipeline always uses the multi-GPU infrastructure, '
                             'even with --num-gpus 1.')

    # HTML export
    parser.add_argument('--samples-per-page', type=int, default=300)
    parser.add_argument('--max-html-samples', type=int, default=0,
                        help='Maximum HTML samples to keep in memory (0=unlimited). '
                             'For full runs with 500K+ cells, set to e.g. 5000 to avoid OOM from base64 crop accumulation.')

    # Server options
    parser.add_argument('--serve', action='store_true', default=False,
                        help='Start HTTP server and wait for Ctrl+C (foreground mode)')
    parser.add_argument('--serve-background', action='store_true', default=True,
                        help='Start HTTP server in background and exit (default: True)')
    parser.add_argument('--no-serve', action='store_true',
                        help='Do not start server after processing')

    # Multi-node sharding
    parser.add_argument('--tile-shard', type=str, default=None,
                        help='Tile shard specification as INDEX/TOTAL (e.g. "0/4" = shard 0 of 4). '
                             'Round-robin assignment: tile i goes to shard i%%TOTAL. '
                             'Implies --detection-only (skips dedup/HTML/CSV).')
    parser.add_argument('--detection-only', action='store_true',
                        help='Skip dedup, HTML generation, and CSV export after tile processing. '
                             'Useful for multi-node runs where a separate merge step handles post-processing.')
    parser.add_argument('--merge-shards', action='store_true', default=False,
                        help='Merge multi-node shard outputs: load all tile detections, dedup, '
                             'generate HTML+CSV. Auto-enabled when --resume finds shard manifests. '
                             'Uses checkpoints so crashes can be resumed.')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for np.random before tissue calibration and tile sampling. '
                             'Ensures all nodes get the same tile list (default: 42).')
    parser.add_argument('--port', type=int, default=8081,
                        help='Port for HTTP server (default: 8081)')
    parser.add_argument('--stop-server', action='store_true',
                        help='Stop any running background server and exit')
    parser.add_argument('--server-status', action='store_true',
                        help='Show status of running server (including public URL) and exit')

    return parser


def postprocess_args(args, parser):
    """Apply cell-type-dependent defaults, parse compound args, validate.

    Args:
        args: Parsed args from parser.parse_args()
        parser: The ArgumentParser (for calling parser.error())

    Returns:
        args: Modified args namespace
    """
    # Cell-type-dependent defaults for output-dir and channel
    if args.output_dir is None:
        args.output_dir = str(Path.cwd() / 'output')
    if args.channel is None:
        if args.cell_type == 'nmj':
            args.channel = 1
        elif args.cell_type == 'islet':
            if getattr(args, 'nuclei_only', False):
                args.channel = getattr(args, 'nuclear_channel', 4)
            else:
                args.channel = getattr(args, 'membrane_channel', 1)
        elif args.cell_type == 'tissue_pattern':
            # Primary channel = first detection channel (for tissue loading)
            if not getattr(args, 'tp_detection_channels', None):
                parser.error("--tp-detection-channels is required for tissue_pattern cell type")
            try:
                args.channel = int(args.tp_detection_channels.split(',')[0])
            except (ValueError, IndexError):
                parser.error(f"--tp-detection-channels: first entry must be integer, got '{args.tp_detection_channels}'")
        elif args.cell_type == 'cell' and args.cellpose_input_channels:
            try:
                args.channel = int(args.cellpose_input_channels.split(',')[0])
            except (ValueError, IndexError):
                parser.error(f"--cellpose-input-channels: first entry must be integer, got '{args.cellpose_input_channels}'")
            # 2-channel Cellpose needs both channels loaded into shared memory
            args.all_channels = True
        else:
            args.channel = 0

    # Handle --cell-type islet: auto-enable all-channels, dedup by area (largest wins)
    if args.cell_type == 'islet':
        args.all_channels = True
        # Parse --islet-display-channels into list of ints
        args.islet_display_chs = [int(x.strip()) for x in args.islet_display_channels.split(',')]
        # Parse --islet-marker-channels into dict: {name: channel_index}
        args.islet_marker_map = {}
        for pair in args.islet_marker_channels.split(','):
            pair = pair.strip()
            if ':' not in pair:
                parser.error(f"--islet-marker-channels: each entry must be NAME:CHANNEL, got '{pair}'")
            name, ch = pair.split(':', 1)
            try:
                args.islet_marker_map[name.strip()] = int(ch.strip())
            except ValueError:
                parser.error(f"--islet-marker-channels: channel must be integer, got '{ch.strip()}' in '{pair}'")

    # Handle --cell-type tissue_pattern: auto-enable all-channels, parse display channels
    if args.cell_type == 'tissue_pattern':
        args.all_channels = True
        args.tp_display_channels_list = [int(x) for x in args.tp_display_channels.split(',')]

    # Handle --multi-marker: automatically enable dependent flags
    if getattr(args, 'multi_marker', False):
        if args.cell_type != 'vessel':
            parser.error("--multi-marker is only valid with --cell-type vessel")
        # Auto-enable all-channels and parallel-detection
        args.all_channels = True
        args.parallel_detection = True
        # Note: logger not available yet, will log in run_pipeline()

    # Auto-detect number of GPUs if not specified
    if args.num_gpus is None:
        try:
            args.num_gpus = max(1, torch.cuda.device_count())
        except Exception:
            args.num_gpus = 1

    # --multi-gpu is always True now (kept for backward compatibility)
    args.multi_gpu = True

    # Handle --tile-shard: parse "INDEX/TOTAL" into tuple, implies --detection-only
    if args.tile_shard is not None:
        try:
            parts = args.tile_shard.split('/')
            shard_idx, shard_total = int(parts[0]), int(parts[1])
            if shard_idx < 0 or shard_idx >= shard_total or shard_total < 1:
                parser.error(f"--tile-shard: INDEX must be 0..TOTAL-1, got {shard_idx}/{shard_total}")
            args.tile_shard = (shard_idx, shard_total)
        except (ValueError, IndexError):
            parser.error(f"--tile-shard must be INDEX/TOTAL (e.g. '0/4'), got '{args.tile_shard}'")
        args.detection_only = True  # sharding implies detection-only
        args.no_serve = True  # no server for detection shards
        if not args.resume and not args.resume_from:
            print("WARNING: --tile-shard without --resume: each node will create its own directory. "
                  "Use --resume <shared-dir> so all shards write to the same location.", flush=True)

    # Handle --resume: also set resume_from for multiscale backward compat
    if args.resume:
        if not Path(args.resume).exists():
            parser.error(f"--resume directory does not exist: {args.resume}")
        # Set resume_from so multiscale checkpoint logic also picks it up
        if args.resume_from is None:
            args.resume_from = args.resume
        # Auto-detect shard manifests -> enable merge-shards
        shard_manifests = list(Path(args.resume).glob('shard_*_manifest.json'))
        if shard_manifests and not args.merge_shards:
            print(f"Auto-detected {len(shard_manifests)} shard manifests -- enabling --merge-shards", flush=True)
            args.merge_shards = True

    # --merge-shards requires --resume
    if args.merge_shards and not args.resume:
        parser.error("--merge-shards requires --resume <shared-output-dir>")

    return args
