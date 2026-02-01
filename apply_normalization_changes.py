#!/usr/bin/env python3
"""
Apply normalization changes to run_unified_FAST.py
"""

from pathlib import Path

def apply_changes():
    script_path = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/run_unified_FAST.py")

    # Read current content
    with open(script_path, 'r') as f:
        lines = f.readlines()

    # 1. Add import after line 27 (after percentile_normalize import)
    import_line = 26  # 0-indexed, so line 27
    new_import = """from segmentation.preprocessing.stain_normalization import (
    percentile_normalize_rgb,
    compute_global_percentiles,
    normalize_to_percentiles
)
"""

    # Find the right place after the imports from segmentation.io.html_export
    for i, line in enumerate(lines):
        if 'from segmentation.io.html_export import' in line:
            # Find the end of this import block
            j = i
            while j < len(lines) and (lines[j].strip().endswith(',') or lines[j].strip().endswith(')')):
                j += 1
            import_line = j + 1
            break

    lines.insert(import_line, new_import)

    # 2. Add arguments to parser
    # Find the argument parser section
    for i, line in enumerate(lines):
        if '--cleanup-masks' in line:
            # Find the next parser.add_argument line
            j = i + 1
            while j < len(lines) and 'parser.add_argument' not in lines[j]:
                j += 1
            # Insert before the next argument
            norm_args = """    parser.add_argument('--normalize-slides', action='store_true',
                        help='Apply cross-slide intensity normalization (recommended for multi-slide batches)')
    parser.add_argument('--norm-percentile-low', type=float, default=1.0,
                        help='Lower percentile for normalization (default: 1.0)')
    parser.add_argument('--norm-percentile-high', type=float, default=99.0,
                        help='Upper percentile for normalization (default: 99.0)')
"""
            lines.insert(j, norm_args)
            break

    # 3. Add normalization logic after Phase 1
    for i, line in enumerate(lines):
        if 'After Phase 1 (all slides in RAM)' in line:
            # Insert normalization code after log_memory_status call
            j = i + 1
            normalization_code = """
        # NORMALIZATION: Compute global percentiles and normalize all slides
        if args.normalize_slides and len(czi_paths) > 1:
            logger.info(f"\\n{'='*70}")
            logger.info("CROSS-SLIDE NORMALIZATION")
            logger.info(f"{'='*70}")
            logger.info(f"Computing global percentiles (P{args.norm_percentile_low}-P{args.norm_percentile_high}) from {len(slide_loaders)} slides...")

            # Collect channel data from all slides for global stats
            all_channel_data = []
            for slide_name, loader in slide_loaders.items():
                channel_data = loader.get_channel_data(channel)
                if channel_data is not None:
                    all_channel_data.append(channel_data)

            # Compute global target percentiles
            target_low, target_high = compute_global_percentiles(
                all_channel_data,
                p_low=args.norm_percentile_low,
                p_high=args.norm_percentile_high,
                n_samples=50000  # Sample 50k pixels per slide
            )

            logger.info(f"  Global target range:")
            logger.info(f"    R: [{target_low[0]:.1f}, {target_high[0]:.1f}]")
            logger.info(f"    G: [{target_low[1]:.1f}, {target_high[1]:.1f}]")
            logger.info(f"    B: [{target_low[2]:.1f}, {target_high[2]:.1f}]")

            # Normalize each slide
            for slide_name, loader in slide_loaders.items():
                logger.info(f"  Normalizing {slide_name}...")
                channel_data = loader.get_channel_data(channel)

                if channel_data is not None:
                    # Show before stats
                    before_mean = channel_data.mean(axis=(0,1))

                    # Normalize
                    normalized = normalize_to_percentiles(
                        channel_data,
                        target_low,
                        target_high,
                        p_low=args.norm_percentile_low,
                        p_high=args.norm_percentile_high
                    )

                    # Update loader's channel data
                    loader.channel_data = normalized

                    # Show after stats
                    after_mean = normalized.mean(axis=(0,1))
                    logger.info(f"    Before: RGB=({before_mean[0]:.1f}, {before_mean[1]:.1f}, {before_mean[2]:.1f})")
                    logger.info(f"    After:  RGB=({after_mean[0]:.1f}, {after_mean[1]:.1f}, {after_mean[2]:.1f})")

            logger.info("Cross-slide normalization complete!")
            log_memory_status("After normalization")

"""
            lines.insert(j, normalization_code)
            break

    # Write back
    with open(script_path, 'w') as f:
        f.writelines(lines)

    print("✓ Normalization support added to run_unified_FAST.py")
    print("\nTo use:")
    print("  Add --normalize-slides flag when running segmentation")
    print("  Example: python run_unified_FAST.py --czi-paths *.czi --output-dir output --normalize-slides")

if __name__ == "__main__":
    apply_changes()
