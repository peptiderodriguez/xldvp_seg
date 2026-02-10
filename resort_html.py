#!/usr/bin/env python3
"""
Re-sort existing HTML annotation pages by area (ascending) without reloading CZI.

Parses cards from existing HTML pages, extracts area from card-stats,
sorts all cards, and rewrites pages with the new order.

Usage:
    python resort_html.py --html-dir /path/to/html --sort-by area --sort-order asc
"""

import argparse
import re
from pathlib import Path
from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def extract_cards_from_html(html_path):
    """Extract card HTML blocks and their area from an HTML page."""
    with open(html_path) as f:
        html = f.read()

    # Extract each card div block (card div contains nested divs, ends with 3 closing divs)
    # Pattern: <div class="card" id="UID" ...> ... </div>\n</div>\n</div>
    card_pattern = re.compile(
        r'(<div\s+class="card"\s+id="([^"]+)".*?</div>\s*</div>\s*</div>)',
        re.DOTALL
    )

    cards = []
    for match in card_pattern.finditer(html):
        card_html = match.group(1)
        uid = match.group(2)

        # Extract area from card-stats: "113.8 µm²" or "113.8 &micro;m&sup2;"
        area_match = re.search(r'([\d.]+)\s*(?:&micro;m&sup2;|µm²)', card_html)
        area_um2 = float(area_match.group(1)) if area_match else 0.0

        # Extract pixel area too
        px_match = re.search(r'(\d+)\s*px', card_html)
        area_px = int(px_match.group(1)) if px_match else 0

        cards.append({
            'uid': uid,
            'html': card_html,
            'area_um2': area_um2,
            'area_px': area_px,
        })

    return cards, html


def extract_page_skeleton(html):
    """Extract the HTML page structure (header, scripts, etc.) without the card grid content."""
    # Find the grid div that contains cards
    grid_start = re.search(r'<div\s+class="grid"[^>]*>', html)
    grid_end_pattern = re.compile(r'</div>\s*</div>\s*<script>', re.DOTALL)
    # Find the </div> that closes the grid, followed by </div> closing the container, then <script>
    # Actually let's find the section between grid open and the script tag
    return grid_start


def resort_html(html_dir, sort_by='area', sort_order='asc'):
    """Re-sort all HTML pages by extracting cards and rewriting."""
    html_dir = Path(html_dir)

    # Find all page files
    page_files = sorted(html_dir.glob('*_page_*.html'),
                        key=lambda p: int(re.search(r'_page_(\d+)', p.name).group(1)))

    if not page_files:
        logger.error(f"No page files found in {html_dir}")
        return

    logger.info(f"Found {len(page_files)} HTML pages")

    # Extract all cards from all pages
    all_cards = []
    for pf in page_files:
        cards, _ = extract_cards_from_html(pf)
        all_cards.extend(cards)
        logger.info(f"  {pf.name}: {len(cards)} cards")

    logger.info(f"Total cards (raw): {len(all_cards)}")

    # Deduplicate by UID (keep last occurrence — from the most recent page)
    seen = {}
    for card in all_cards:
        seen[card['uid']] = card
    all_cards = list(seen.values())
    logger.info(f"After dedup: {len(all_cards)}")

    # Sort
    reverse = (sort_order == 'desc')
    if sort_by == 'area':
        all_cards.sort(key=lambda c: c['area_um2'], reverse=reverse)
    elif sort_by == 'area_px':
        all_cards.sort(key=lambda c: c['area_px'], reverse=reverse)
    else:
        all_cards.sort(key=lambda c: c.get(sort_by, 0), reverse=reverse)

    logger.info(f"Sorted by {sort_by} ({sort_order})")
    if all_cards:
        logger.info(f"  First: {all_cards[0]['uid']} ({all_cards[0]['area_um2']:.1f} µm²)")
        logger.info(f"  Last: {all_cards[-1]['uid']} ({all_cards[-1]['area_um2']:.1f} µm²)")

    # Read the first page to get the template structure
    with open(page_files[0]) as f:
        template_html = f.read()

    # Figure out samples per page from existing pages
    first_page_cards, _ = extract_cards_from_html(page_files[0])
    samples_per_page = len(first_page_cards)
    n_pages = (len(all_cards) + samples_per_page - 1) // samples_per_page
    logger.info(f"Rewriting {n_pages} pages ({samples_per_page} per page)")

    # Extract the cell type prefix from filenames
    cell_type = re.match(r'([^_]+)_page_', page_files[0].name).group(1)

    # For each page: replace the grid contents with re-sorted cards
    for page_num in range(1, n_pages + 1):
        start_idx = (page_num - 1) * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(all_cards))
        page_cards = all_cards[start_idx:end_idx]

        # Read existing page (or first page as template if we have fewer pages now)
        page_file = html_dir / f'{cell_type}_page_{page_num}.html'
        if page_file.exists():
            with open(page_file) as f:
                page_html = f.read()
        else:
            page_html = template_html

        # Replace grid contents: find grid div, replace everything inside
        grid_content = '\n                '.join(c['html'] for c in page_cards)

        # Pattern: match from <div class="grid"> to the closing </div> before the next sibling
        new_html = re.sub(
            r'(<div\s+class="grid"[^>]*>).*?(</div>\s*</div>\s*(?:<script>|<div\s+class="footer"))',
            lambda m: m.group(1) + '\n                ' + grid_content + '\n            </div>\n        </div>\n        ' + m.group(2).lstrip(),
            page_html,
            count=1,
            flags=re.DOTALL
        )

        # Update page navigation if page count changed
        # Update "Page X/Y" display
        new_html = re.sub(
            r'Page \d+/\d+',
            f'Page {page_num}/{n_pages}',
            new_html
        )

        with open(page_file, 'w') as f:
            f.write(new_html)
        logger.info(f"  Wrote {page_file.name}: {len(page_cards)} cards")

    # Remove extra pages if we now have fewer
    for old_page in page_files:
        page_num = int(re.search(r'_page_(\d+)', old_page.name).group(1))
        if page_num > n_pages:
            old_page.unlink()
            logger.info(f"  Removed extra page: {old_page.name}")

    logger.info(f"Done! {len(all_cards)} cards re-sorted by {sort_by} ({sort_order})")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Re-sort HTML annotation pages without reloading CZI')
    parser.add_argument('--html-dir', required=True, help='Path to HTML directory')
    parser.add_argument('--sort-by', default='area', choices=['area', 'area_px'], help='Sort field (default: area)')
    parser.add_argument('--sort-order', default='asc', choices=['asc', 'desc'], help='Sort order (default: asc)')
    args = parser.parse_args()

    resort_html(args.html_dir, sort_by=args.sort_by, sort_order=args.sort_order)


if __name__ == '__main__':
    main()
