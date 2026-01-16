#!/usr/bin/env python3
"""Add Clear Page and Clear All buttons to MK and HSPC annotation pages."""

import re
from pathlib import Path
from glob import glob

def add_clear_buttons(html_content):
    """Add clear buttons to the stats row and JavaScript functions."""

    # Add CSS for clear buttons if not present
    if '.clear-btn' not in html_content:
        css_addition = '''
        .clear-btn { padding: 4px 12px; background: #1a1a1a; color: #888; border: 1px solid #333; cursor: pointer; font-size: 0.8em; margin-left: 10px; }
        .clear-btn:hover { background: #2a1a1a; color: #a44; border-color: #a44; }'''

        # Insert before closing </style>
        html_content = html_content.replace('</style>', css_addition + '\n    </style>')

    # Add clear buttons after the stats row if not present
    if 'clearPage()' not in html_content:
        # Find the stats-row section and add buttons after global stats
        # Look for the pattern: global-negative-count followed by closing div
        pattern = r'(<span id="global-negative-count">\d+</span>\s*</div>\s*</div>)'
        replacement = r'\1\n            <button class="clear-btn" onclick="clearPage()">Clear Page</button>\n            <button class="clear-btn" onclick="clearAll()">Clear All</button>'
        html_content = re.sub(pattern, replacement, html_content)

    # Add JavaScript functions if not present
    if 'function clearPage()' not in html_content:
        js_functions = '''
        function clearPage() {
            if (!confirm('Clear all annotations on this page?')) return;
            document.querySelectorAll('.card').forEach(card => {
                card.dataset.label = -1;
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            });
            localStorage.removeItem(STORAGE_KEY);
            updateStats();
        }

        function clearAll() {
            if (!confirm('Clear ALL annotations across all ' + TOTAL_PAGES + ' pages?')) return;
            for (let i = 1; i <= TOTAL_PAGES; i++) {
                localStorage.removeItem(CELL_TYPE + '_labels_page' + i);
            }
            document.querySelectorAll('.card').forEach(card => {
                card.dataset.label = -1;
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            });
            updateStats();
        }
'''
        # Insert before loadAnnotations() call at the end
        html_content = html_content.replace('loadAnnotations();', js_functions + '\n        loadAnnotations();')

    return html_content

def process_files(pattern):
    """Process all files matching the pattern."""
    files = sorted(glob(pattern))
    print(f"Processing {len(files)} files matching {pattern}")

    for filepath in files:
        with open(filepath, 'r') as f:
            content = f.read()

        new_content = add_clear_buttons(content)

        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"  Updated: {Path(filepath).name}")

def main():
    docs_dir = Path('/home/dude/code/xldvp_seg_repo/docs')

    # Process MK pages
    process_files(str(docs_dir / 'mk_page*.html'))

    # Process HSPC pages
    process_files(str(docs_dir / 'hspc_page*.html'))

    print("\nDone!")

if __name__ == '__main__':
    main()
