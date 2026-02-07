"""
Unified HTML page generator for cell annotation interfaces.

Provides a class-based API for generating HTML annotation pages with:
- Configurable localStorage strategies (page-specific, global, experiment)
- Dark theme styling
- Keyboard navigation (Y/N/U for labeling, arrows for navigation)
- Local + global annotation statistics
- JSON export functionality
- Custom stat formatters

Example usage:
    generator = HTMLPageGenerator(
        cell_type='mk',
        experiment_name='2025_batch1',
        storage_strategy='experiment',
        samples_per_page=300
    )

    # Register custom formatter
    generator.register_formatter('custom_stat', lambda v: f'{v:.3f}')

    # Export all samples to HTML files
    generator.export_to_html(samples, '/path/to/output')
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union


# Type aliases
StorageStrategy = Literal['page-specific', 'global', 'experiment']
FormatterFunc = Callable[[Any], str]


class HTMLPageGenerator:
    """
    Unified HTML page generator for cell annotation interfaces.

    Generates dark-themed HTML pages with:
    - Grid-based card layout for cell images
    - Keyboard navigation and labeling
    - localStorage-based annotation persistence
    - Local (page) and global statistics
    - JSON export functionality

    Attributes:
        cell_type: Identifier for the cell type (e.g., 'mk', 'hspc', 'nmj')
        experiment_name: Optional experiment name for localStorage isolation
        storage_strategy: How to organize localStorage keys
        samples_per_page: Number of samples per HTML page
        title: Optional custom title for pages
        formatters: Dict mapping stat keys to formatter functions
    """

    # Default formatters for common statistics
    DEFAULT_FORMATTERS: Dict[str, FormatterFunc] = {
        'area_um2': lambda v: f"{v:.1f} &micro;m&sup2;",
        'area_px': lambda v: f"{v:.0f} px",
        'confidence': lambda v: f"{v * 100:.0f}%",
        'elongation': lambda v: f"elong: {v:.2f}",
    }

    # Dark theme color palette
    COLORS = {
        'bg_primary': '#0a0a0a',
        'bg_secondary': '#111',
        'bg_tertiary': '#1a1a1a',
        'border': '#333',
        'text_primary': '#ddd',
        'text_secondary': '#888',
        'text_tertiary': '#555',
        'positive': '#4a4',
        'negative': '#a44',
        'unsure': '#aa4',
        'export': '#44a',
        'card_yes_bg': '#0f130f',
        'card_no_bg': '#130f0f',
        'card_unsure_bg': '#13130f',
    }

    def __init__(
        self,
        cell_type: str,
        experiment_name: Optional[str] = None,
        storage_strategy: StorageStrategy = 'global',
        samples_per_page: int = 300,
        title: Optional[str] = None,
    ) -> None:
        """
        Initialize the HTML page generator.

        Args:
            cell_type: Identifier for the cell type (e.g., 'mk', 'hspc', 'nmj').
                      Used in localStorage keys and file naming.
            experiment_name: Optional experiment identifier for localStorage isolation.
                           Required when storage_strategy is 'experiment'.
            storage_strategy: How to organize localStorage keys:
                - 'page-specific': Separate key per page (legacy behavior)
                - 'global': Single key for all annotations of this cell type
                - 'experiment': Key per experiment (requires experiment_name)
            samples_per_page: Number of samples to display per HTML page.
            title: Optional custom title. Defaults to cell_type.upper().

        Raises:
            ValueError: If storage_strategy is 'experiment' but experiment_name is None.
        """
        if storage_strategy == 'experiment' and experiment_name is None:
            raise ValueError(
                "experiment_name is required when storage_strategy is 'experiment'"
            )

        self.cell_type = cell_type
        self.experiment_name = experiment_name
        self.storage_strategy = storage_strategy
        self.samples_per_page = samples_per_page
        self.title = title or cell_type.upper()

        # Initialize formatters with defaults
        self.formatters: Dict[str, FormatterFunc] = dict(self.DEFAULT_FORMATTERS)

    def register_formatter(
        self,
        stat_key: str,
        formatter_func: FormatterFunc,
    ) -> None:
        """
        Register a custom formatter for a statistic key.

        Args:
            stat_key: The key in sample['stats'] to format.
            formatter_func: Function that takes a value and returns formatted string.
                          May include HTML entities.

        Example:
            generator.register_formatter('diameter', lambda v: f'{v:.2f} um')
            generator.register_formatter('score', lambda v: f'{v*100:.1f}%')
        """
        self.formatters[stat_key] = formatter_func

    def get_storage_key(self, page_num: Optional[int] = None) -> str:
        """
        Get the localStorage key based on the configured strategy.

        Args:
            page_num: Page number (required for 'page-specific' strategy).

        Returns:
            The localStorage key string.

        Raises:
            ValueError: If strategy is 'page-specific' and page_num is None.
        """
        if self.storage_strategy == 'page-specific':
            if page_num is None:
                raise ValueError(
                    "page_num is required for 'page-specific' storage strategy"
                )
            return f'{self.cell_type}_labels_page{page_num}'
        elif self.storage_strategy == 'global':
            return f'{self.cell_type}_annotations'
        elif self.storage_strategy == 'experiment':
            return f'{self.cell_type}_{self.experiment_name}_annotations'
        else:
            # Fallback to global
            return f'{self.cell_type}_annotations'

    def _generate_css(self) -> str:
        """
        Generate the dark theme CSS styles.

        Returns:
            CSS string for inclusion in HTML <style> tags.
        """
        c = self.COLORS
        return f'''
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: {c['bg_primary']}; color: {c['text_primary']}; }}

        .header {{
            background: {c['bg_secondary']};
            padding: 12px 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            border-bottom: 1px solid {c['border']};
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .header h1 {{
            font-size: 1.2em;
            font-weight: normal;
        }}

        .nav-buttons {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .nav-btn {{
            padding: 8px 15px;
            background: {c['bg_tertiary']};
            border: 1px solid {c['border']};
            color: {c['text_primary']};
            text-decoration: none;
            cursor: pointer;
            font-family: monospace;
        }}

        .nav-btn:hover {{
            background: #222;
        }}

        .page-info {{
            padding: 8px 15px;
            color: {c['text_secondary']};
        }}

        .stats-row {{
            display: flex;
            gap: 20px;
            font-size: 0.85em;
            flex-wrap: wrap;
            align-items: center;
        }}

        .stats-group {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}

        .stats-label {{
            color: {c['text_secondary']};
            font-size: 0.9em;
        }}

        .stat {{
            padding: 4px 10px;
            background: {c['bg_tertiary']};
            border: 1px solid {c['border']};
        }}

        .stat.positive {{
            border-left: 3px solid {c['positive']};
        }}

        .stat.negative {{
            border-left: 3px solid {c['negative']};
        }}

        .content {{
            padding: 15px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
        }}

        .card {{
            background: {c['bg_secondary']};
            border: 2px solid {c['border']};
            overflow: hidden;
            transition: border-color 0.2s;
        }}

        .card.selected {{
            box-shadow: 0 0 0 3px #fff;
        }}

        .card.labeled-yes {{
            border-color: {c['positive']} !important;
            background: {c['card_yes_bg']} !important;
        }}

        .card.labeled-no {{
            border-color: {c['negative']} !important;
            background: {c['card_no_bg']} !important;
        }}

        .card.labeled-unsure {{
            border-color: {c['unsure']} !important;
            background: {c['card_unsure_bg']} !important;
        }}

        .card-img-container {{
            width: 100%;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
        }}

        .card img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}

        .card-info {{
            padding: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid {c['border']};
            gap: 10px;
        }}

        .card-meta {{
            flex: 1;
            min-width: 0;
        }}

        .card-id {{
            font-size: 0.75em;
            color: {c['text_secondary']};
            word-break: break-all;
        }}

        .card-stats {{
            font-size: 0.8em;
            color: #aaa;
            margin-top: 3px;
        }}

        .buttons {{
            display: flex;
            gap: 5px;
            flex-shrink: 0;
        }}

        .btn {{
            padding: 6px 12px;
            border: 1px solid {c['border']};
            background: {c['bg_tertiary']};
            color: {c['text_primary']};
            cursor: pointer;
            font-family: monospace;
            font-size: 0.85em;
        }}

        .btn:hover {{
            background: #222;
        }}

        .btn-yes {{
            border-color: {c['positive']};
            color: {c['positive']};
        }}

        .btn-no {{
            border-color: {c['negative']};
            color: {c['negative']};
        }}

        .btn-unsure {{
            border-color: {c['unsure']};
            color: {c['unsure']};
        }}

        .btn-export {{
            border-color: {c['export']};
            color: {c['export']};
        }}

        .keyboard-hint {{
            text-align: center;
            padding: 15px;
            color: {c['text_tertiary']};
            font-size: 0.85em;
            border-top: 1px solid #222;
        }}

        .footer {{
            background: {c['bg_secondary']};
            padding: 15px;
            border-top: 1px solid {c['border']};
            display: flex;
            justify-content: center;
            gap: 10px;
        }}
        '''

    def _generate_js(self, total_pages: int = 1) -> str:
        """
        Generate the unified JavaScript for annotation handling.

        Includes:
        - localStorage management based on configured strategy
        - Keyboard navigation (arrow keys)
        - Label shortcuts (Y/N/U)
        - Local and global statistics calculation
        - JSON export functionality
        - Clear page functionality

        Args:
            total_pages: Total number of pages for navigation.

        Returns:
            JavaScript code string.
        """
        storage_key = self.get_storage_key(page_num=1)  # Base key for global/experiment

        return f'''
        const CELL_TYPE = '{self.cell_type}';
        const EXPERIMENT_NAME = '{self.experiment_name or ""}';
        const TOTAL_PAGES = {total_pages};
        const STORAGE_STRATEGY = '{self.storage_strategy}';

        // Get storage key based on strategy
        function getStorageKey(pageNum) {{
            if (STORAGE_STRATEGY === 'page-specific') {{
                return CELL_TYPE + '_labels_page' + pageNum;
            }} else if (STORAGE_STRATEGY === 'experiment' && EXPERIMENT_NAME) {{
                return CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations';
            }} else {{
                return CELL_TYPE + '_annotations';
            }}
        }}

        const STORAGE_KEY = getStorageKey(typeof PAGE_NUM !== 'undefined' ? PAGE_NUM : 1);

        let labels = {{}};
        let selectedIdx = -1;
        const cards = document.querySelectorAll('.card');

        // Load from localStorage
        function loadAnnotations() {{
            try {{
                const saved = localStorage.getItem(STORAGE_KEY);
                if (saved) labels = JSON.parse(saved);
            }} catch(e) {{ console.error(e); }}

            // Apply to cards
            cards.forEach((card, i) => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    applyLabelToCard(card, labels[uid]);
                }}
            }});

            updateStats();
        }}

        function applyLabelToCard(card, label) {{
            card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            card.dataset.label = label;
            if (label === 1) card.classList.add('labeled-yes');
            else if (label === 0) card.classList.add('labeled-no');
            else if (label === 2) card.classList.add('labeled-unsure');
        }}

        function setLabel(uid, label, autoAdvance = false) {{
            // Toggle off if same label
            if (labels[uid] === label) {{
                delete labels[uid];
                const card = document.getElementById(uid);
                if (card) {{
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                    card.dataset.label = -1;
                }}
            }} else {{
                labels[uid] = label;
                const card = document.getElementById(uid);
                if (card) applyLabelToCard(card, label);
            }}

            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
            updateStats();

            if (autoAdvance && selectedIdx >= 0 && selectedIdx < cards.length - 1) {{
                selectCard(selectedIdx + 1);
            }}
        }}

        function updateStats() {{
            let localYes = 0, localNo = 0, localUnsure = 0;
            let globalYes = 0, globalNo = 0, globalUnsure = 0;

            // Count current page
            cards.forEach(card => {{
                const uid = card.id;
                if (labels[uid] === 1) localYes++;
                else if (labels[uid] === 0) localNo++;
                else if (labels[uid] === 2) localUnsure++;
            }});

            // Count global
            for (const v of Object.values(labels)) {{
                if (v === 1) globalYes++;
                else if (v === 0) globalNo++;
                else if (v === 2) globalUnsure++;
            }}

            // Update DOM if elements exist
            const localYesEl = document.getElementById('localYes');
            const localNoEl = document.getElementById('localNo');
            const globalYesEl = document.getElementById('globalYes');
            const globalNoEl = document.getElementById('globalNo');

            if (localYesEl) localYesEl.textContent = localYes;
            if (localNoEl) localNoEl.textContent = localNo;
            if (globalYesEl) globalYesEl.textContent = globalYes;
            if (globalNoEl) globalNoEl.textContent = globalNo;
        }}

        function selectCard(idx) {{
            cards.forEach(c => c.classList.remove('selected'));
            if (idx >= 0 && idx < cards.length) {{
                selectedIdx = idx;
                cards[idx].classList.add('selected');
                cards[idx].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        function exportAnnotations() {{
            const data = {{
                cell_type: CELL_TYPE,
                experiment_name: EXPERIMENT_NAME || undefined,
                storage_strategy: STORAGE_STRATEGY,
                exported_at: new Date().toISOString(),
                positive: [],
                negative: [],
                unsure: []
            }};

            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1) data.positive.push(uid);
                else if (label === 0) data.negative.push(uid);
                else if (label === 2) data.unsure.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filename = EXPERIMENT_NAME
                ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations.json'
                : CELL_TYPE + '_annotations.json';
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}

        function clearPage() {{
            if (!confirm('Clear annotations on this page?')) return;
            cards.forEach(card => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    delete labels[uid];
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                    card.dataset.label = -1;
                }}
            }});
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
            updateStats();
        }}

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            // Navigation with arrow keys
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
                e.preventDefault();
                selectCard(Math.min(selectedIdx + 1, cards.length - 1));
            }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
                e.preventDefault();
                selectCard(Math.max(selectedIdx - 1, 0));
            }}
            // Labeling with Y/N/U keys
            else if (selectedIdx >= 0) {{
                const uid = cards[selectedIdx].id;
                if (e.key.toLowerCase() === 'y') setLabel(uid, 1, true);
                else if (e.key.toLowerCase() === 'n') setLabel(uid, 0, true);
                else if (e.key.toLowerCase() === 'u') setLabel(uid, 2, true);
            }}
        }});

        // Initialize
        loadAnnotations();
        '''

    def _generate_nav_html(
        self,
        page_num: int,
        total_pages: int,
        page_prefix: str = 'page',
    ) -> str:
        """
        Generate navigation buttons HTML.

        Args:
            page_num: Current page number (1-indexed).
            total_pages: Total number of pages.
            page_prefix: Prefix for page filenames.

        Returns:
            HTML string with navigation buttons.
        """
        nav_parts = ['<div class="nav-buttons">']
        nav_parts.append('<a href="../index.html" class="nav-btn">Home</a>')

        if page_num > 1:
            nav_parts.append(
                f'<a href="{page_prefix}_{page_num - 1}.html" class="nav-btn">Prev</a>'
            )

        nav_parts.append(f'<span class="page-info">Page {page_num} / {total_pages}</span>')

        if page_num < total_pages:
            nav_parts.append(
                f'<a href="{page_prefix}_{page_num + 1}.html" class="nav-btn">Next</a>'
            )

        nav_parts.append('</div>')
        return ''.join(nav_parts)

    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """
        Format statistics dictionary using registered formatters.

        Args:
            stats: Dictionary of stat key -> value.

        Returns:
            Formatted string with stats separated by ' | '.
        """
        parts = []
        for key, value in stats.items():
            if key in self.formatters:
                try:
                    parts.append(self.formatters[key](value))
                except (TypeError, ValueError):
                    # Fallback to string representation
                    parts.append(str(value))
            else:
                # No formatter registered, use default representation
                if isinstance(value, float):
                    parts.append(f"{key}: {value:.2f}")
                else:
                    parts.append(f"{key}: {value}")

        return ' | '.join(parts)

    def _generate_card_html(self, sample: Dict[str, Any]) -> str:
        """
        Generate HTML for a single sample card.

        Args:
            sample: Dictionary with keys:
                - 'uid': Unique identifier for the sample
                - 'image': Base64-encoded image string
                - 'stats': Optional dict of statistics to display
                - 'mime_type': Optional image mime type (default 'jpeg')

        Returns:
            HTML string for the card.
        """
        uid = sample['uid']
        img_b64 = sample['image']
        mime = sample.get('mime_type', 'jpeg')
        stats = sample.get('stats', {})

        stats_str = self._format_stats(stats)

        return f'''
        <div class="card" id="{uid}" data-label="-1">
            <div class="card-img-container">
                <img src="data:image/{mime};base64,{img_b64}" alt="{uid}">
            </div>
            <div class="card-info">
                <div class="card-meta">
                    <div class="card-id">{uid}</div>
                    <div class="card-stats">{stats_str}</div>
                </div>
                <div class="buttons">
                    <button class="btn btn-yes" onclick="setLabel('{uid}', 1)">Y</button>
                    <button class="btn btn-unsure" onclick="setLabel('{uid}', 2)">?</button>
                    <button class="btn btn-no" onclick="setLabel('{uid}', 0)">N</button>
                </div>
            </div>
        </div>
'''

    def generate_index_html(
        self,
        total_samples: int,
        total_pages: int,
        extra_stats: Optional[Dict[str, Any]] = None,
        page_prefix: str = 'page',
        subtitle: Optional[str] = None,
    ) -> str:
        """
        Generate the index/landing page HTML.

        Args:
            total_samples: Total number of samples across all pages.
            total_pages: Total number of annotation pages.
            extra_stats: Optional dict of additional statistics to display.
                        Keys are labels, values are displayed values.
            page_prefix: Prefix for page filenames.
            subtitle: Optional subtitle for the page.

        Returns:
            Complete HTML string for the index page.
        """
        if subtitle is None:
            subtitle = "Cell Annotation Interface"

        # Build extra stats HTML
        extra_stats_html = ''
        if extra_stats:
            for label, value in extra_stats.items():
                extra_stats_html += f'''
            <div class="stat">
                <span>{label}</span>
                <span class="number">{value}</span>
            </div>'''

        # Build storage key for export
        storage_key_js = self._get_storage_key_js()

        c = self.COLORS
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: {c['bg_primary']}; color: {c['text_primary']}; padding: 20px; }}

        .header {{
            background: {c['bg_secondary']};
            padding: 30px;
            border: 1px solid {c['border']};
            margin-bottom: 20px;
            text-align: center;
        }}

        h1 {{
            font-size: 1.5em;
            font-weight: normal;
            margin-bottom: 10px;
        }}

        .subtitle {{
            color: {c['text_secondary']};
            margin-bottom: 20px;
        }}

        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 25px 0;
            flex-wrap: wrap;
        }}

        .stat {{
            padding: 15px 30px;
            background: {c['bg_tertiary']};
            border: 1px solid {c['border']};
            text-align: center;
        }}

        .stat .number {{
            display: block;
            font-size: 2em;
            margin-top: 10px;
            color: {c['positive']};
        }}

        .section {{
            margin: 30px 0;
            text-align: center;
        }}

        .btn {{
            padding: 15px 40px;
            background: {c['bg_tertiary']};
            border: 1px solid {c['positive']};
            color: {c['positive']};
            cursor: pointer;
            font-family: monospace;
            font-size: 1.1em;
            text-decoration: none;
            display: inline-block;
            margin: 10px;
        }}

        .btn:hover {{
            background: {c['card_yes_bg']};
        }}

        .btn-export {{
            border-color: {c['export']};
            color: {c['export']};
        }}

        .btn-export:hover {{
            background: #0f0f13;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p class="subtitle">{subtitle}</p>
        <div class="stats">
            <div class="stat">
                <span>Total Samples</span>
                <span class="number">{total_samples:,}</span>
            </div>
            <div class="stat">
                <span>Pages</span>
                <span class="number">{total_pages}</span>
            </div>
            {extra_stats_html}
        </div>
    </div>

    <div class="section">
        <a href="{page_prefix}_1.html" class="btn">Start Review</a>
    </div>

    <div class="section">
        <button class="btn btn-export" onclick="exportAnnotations()">Export Annotations</button>
    </div>

    <script>
        const CELL_TYPE = '{self.cell_type}';
        const EXPERIMENT_NAME = '{self.experiment_name or ""}';
        const STORAGE_STRATEGY = '{self.storage_strategy}';

        {storage_key_js}

        function exportAnnotations() {{
            const stored = localStorage.getItem(STORAGE_KEY);
            const labels = stored ? JSON.parse(stored) : {{}};

            const data = {{
                cell_type: CELL_TYPE,
                experiment_name: EXPERIMENT_NAME || undefined,
                storage_strategy: STORAGE_STRATEGY,
                exported_at: new Date().toISOString(),
                positive: [],
                negative: [],
                unsure: []
            }};

            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1) data.positive.push(uid);
                else if (label === 0) data.negative.push(uid);
                else if (label === 2) data.unsure.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filename = EXPERIMENT_NAME
                ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations.json'
                : CELL_TYPE + '_annotations.json';
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>'''

    def _get_storage_key_js(self) -> str:
        """
        Generate JavaScript code to compute the storage key.

        Returns:
            JavaScript code that sets STORAGE_KEY variable.
        """
        if self.storage_strategy == 'page-specific':
            return '''
        var _page_num = typeof PAGE_NUM !== 'undefined' ? PAGE_NUM : 1;
        const STORAGE_KEY = CELL_TYPE + '_labels_page' + _page_num;
            '''
        elif self.storage_strategy == 'experiment':
            return f'''
        const STORAGE_KEY = CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations';
            '''
        else:  # global
            return '''
        const STORAGE_KEY = CELL_TYPE + '_annotations';
            '''

    def generate_page_html(
        self,
        samples: List[Dict[str, Any]],
        page_num: int,
        total_pages: int,
        page_prefix: str = 'page',
    ) -> str:
        """
        Generate an annotation page HTML.

        Args:
            samples: List of sample dictionaries. Each should have:
                - 'uid': Unique identifier
                - 'image': Base64-encoded image
                - 'stats': Optional dict of statistics
            page_num: Current page number (1-indexed).
            total_pages: Total number of pages.
            page_prefix: Prefix for page filenames.

        Returns:
            Complete HTML string for the annotation page.
        """
        # Generate navigation
        nav_html = self._generate_nav_html(page_num, total_pages, page_prefix)

        # Generate cards
        cards_html = ''.join(self._generate_card_html(sample) for sample in samples)

        # Page-specific JS variable for storage key calculation
        page_num_js = f'const PAGE_NUM = {page_num};' if self.storage_strategy == 'page-specific' else ''

        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{self.title} - Page {page_num}/{total_pages}</title>
    <style>{self._generate_css()}</style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <h1>{self.title} - Page {page_num}/{total_pages}</h1>
            {nav_html}
        </div>
        <div class="stats-row">
            <div class="stats-group">
                <span class="stats-label">Page:</span>
                <div class="stat positive">Yes: <span id="localYes">0</span></div>
                <div class="stat negative">No: <span id="localNo">0</span></div>
            </div>
            <div class="stats-group">
                <span class="stats-label">Total:</span>
                <div class="stat positive">Yes: <span id="globalYes">0</span></div>
                <div class="stat negative">No: <span id="globalNo">0</span></div>
            </div>
            <button class="btn btn-export" onclick="exportAnnotations()">Export</button>
            <button class="btn" onclick="clearPage()">Clear Page</button>
        </div>
    </div>

    <div class="content">
        <div class="grid">{cards_html}</div>
    </div>

    <div class="keyboard-hint">
        Keyboard: Y=Yes, N=No, U=Unsure, Arrow keys=Navigate
    </div>

    <div class="footer">
        {nav_html}
    </div>

    <script>
        {page_num_js}
        {self._generate_js(total_pages)}
    </script>
</body>
</html>'''

    def export_to_html(
        self,
        samples: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        page_prefix: str = 'page',
        subtitle: Optional[str] = None,
        extra_stats: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> tuple[int, int]:
        """
        Export all samples to paginated HTML files.

        Creates an index page and multiple annotation pages in the output directory.

        Args:
            samples: List of sample dictionaries. Each should have:
                - 'uid': Unique identifier
                - 'image': Base64-encoded image
                - 'stats': Optional dict of statistics
            output_dir: Directory to write HTML files.
            page_prefix: Prefix for page filenames (e.g., 'page' -> 'page_1.html').
            subtitle: Optional subtitle for the index page.
            extra_stats: Optional dict of extra statistics for the index page.
            verbose: Whether to print progress messages.

        Returns:
            Tuple of (total_samples, total_pages).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not samples:
            if verbose:
                print(f"No {self.cell_type} samples to export")
            return 0, 0

        # Paginate samples
        pages = [
            samples[i:i + self.samples_per_page]
            for i in range(0, len(samples), self.samples_per_page)
        ]
        total_pages = len(pages)

        if verbose:
            print(f"Generating {total_pages} {self.cell_type} HTML pages...")

        # Generate annotation pages
        for page_num, page_samples in enumerate(pages, 1):
            html = self.generate_page_html(
                page_samples,
                page_num,
                total_pages,
                page_prefix,
            )

            page_path = output_dir / f"{page_prefix}_{page_num}.html"
            with open(page_path, 'w', encoding='utf-8') as f:
                f.write(html)

            if verbose:
                file_size = page_path.stat().st_size / (1024 * 1024)
                print(f"  Page {page_num}: {len(page_samples)} samples ({file_size:.1f} MB)")

        # Generate index page
        index_html = self.generate_index_html(
            total_samples=len(samples),
            total_pages=total_pages,
            extra_stats=extra_stats,
            page_prefix=page_prefix,
            subtitle=subtitle,
        )

        index_path = output_dir / 'index.html'
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_html)

        if verbose:
            print(f"Export complete: {output_dir}")

        return len(samples), total_pages


# =============================================================================
# MK/HSPC BATCH HTML EXPORT FUNCTIONS
# (Moved from run_unified_FAST.py for consolidation)
# =============================================================================

import base64
import h5py
import json
import logging
import re
from io import BytesIO

import numpy as np
from PIL import Image

# Get logger (may not be initialized if imported standalone)
try:
    from segmentation.utils.logging import get_logger
    _logger = get_logger(__name__)
except ImportError:
    _logger = logging.getLogger(__name__)


def load_samples_from_ram(tiles_dir, slide_image, pixel_size_um, cell_type='mk', max_samples=None):
    """
    Load cell samples from segmentation output, using in-memory slide image.

    This function reads segmentation results from disk and extracts image crops
    from a slide image already loaded into RAM, avoiding repeated disk/network I/O.

    Args:
        tiles_dir: Path to tiles directory (e.g., output/mk/tiles)
        slide_image: numpy array of full slide image (already in RAM)
        pixel_size_um: Pixel size in microns
        cell_type: 'mk' or 'hspc' - affects mask selection
        max_samples: Maximum samples to load (None for all)

    Returns:
        List of sample dicts with image data and metadata
    """
    from segmentation.io.html_export import (
        get_largest_connected_component,
        draw_mask_contour,
        percentile_normalize,
    )

    tiles_dir = Path(tiles_dir)
    if not tiles_dir.exists():
        return []

    samples = []
    tile_dirs = sorted([d for d in tiles_dir.iterdir() if d.is_dir()],
                       key=lambda x: int(x.name))

    for tile_dir in tile_dirs:
        features_file = tile_dir / "features.json"
        seg_file = tile_dir / "segmentation.h5"
        window_file = tile_dir / "window.csv"

        if not all(f.exists() for f in [features_file, seg_file, window_file]):
            continue

        # Load tile window coordinates
        with open(window_file, 'r') as f:
            window_str = f.read().strip()
        try:
            matches = re.findall(r'slice\((\d+),\s*(\d+)', window_str)
            if len(matches) >= 2:
                tile_y1, tile_y2 = int(matches[0][0]), int(matches[0][1])
                tile_x1, tile_x2 = int(matches[1][0]), int(matches[1][1])
            else:
                continue
        except Exception as e:
            _logger.debug(f"Failed to parse tile coordinates from {seg_file.parent.name}: {e}")
            continue

        # Load features
        with open(features_file, 'r') as f:
            tile_features = json.load(f)

        # Load segmentation masks
        with h5py.File(seg_file, 'r') as f:
            masks = f['labels'][0]  # Shape: (H, W)

        # For HSPCs, sort by solidity (higher = more confident/solid shape)
        if cell_type == 'hspc':
            tile_features = sorted(tile_features,
                                   key=lambda x: x['features'].get('solidity', 0),
                                   reverse=True)

        # Extract each cell
        for feat_dict in tile_features:
            det_id = feat_dict['id']
            features = feat_dict['features']
            area_px = features.get('area', 0)
            area_um2 = area_px * (pixel_size_um ** 2)

            try:
                cell_idx = int(det_id.split('_')[1]) + 1
            except Exception as e:
                _logger.debug(f"Failed to parse cell index from {det_id}: {e}")
                continue

            cell_mask = masks == cell_idx
            if not cell_mask.any():
                cell_mask = masks == int(det_id.split('_')[1])
                if not cell_mask.any():
                    continue

            # For MKs, extract only the largest connected component
            if cell_type == 'mk':
                cell_mask = get_largest_connected_component(cell_mask)
                if not cell_mask.any():
                    continue

            ys, xs = np.where(cell_mask)
            if len(ys) == 0:
                continue

            # Calculate mask centroid for centering
            centroid_y = int(np.mean(ys))
            centroid_x = int(np.mean(xs))

            # Calculate mask bounding box
            y1_local, y2_local = ys.min(), ys.max()
            x1_local, x2_local = xs.min(), xs.max()
            mask_h = y2_local - y1_local
            mask_w = x2_local - x1_local

            # Create a centered crop around the mask centroid
            # Crop size should be at least mask size + padding, minimum 300px
            crop_size = max(300, max(mask_h, mask_w) + 100)
            half_size = crop_size // 2

            # Crop bounds centered on centroid
            crop_y1 = max(0, centroid_y - half_size)
            crop_y2 = min(masks.shape[0], centroid_y + half_size)
            crop_x1 = max(0, centroid_x - half_size)
            crop_x2 = min(masks.shape[1], centroid_x + half_size)

            # Read from in-memory slide image (instead of CZI)
            global_y1 = tile_y1 + crop_y1
            global_y2 = tile_y1 + crop_y2
            global_x1 = tile_x1 + crop_x1
            global_x2 = tile_x1 + crop_x2

            # Bounds check
            global_y2 = min(global_y2, slide_image.shape[0])
            global_x2 = min(global_x2, slide_image.shape[1])

            try:
                crop = slide_image[global_y1:global_y2, global_x1:global_x2]
            except Exception as e:
                _logger.debug(f"Failed to extract crop at ({global_x1}, {global_y1}): {e}")
                continue

            if crop is None or crop.size == 0:
                continue

            # Convert to RGB if needed
            if crop.ndim == 2:
                crop = np.stack([crop] * 3, axis=-1)
            elif crop.shape[2] == 4:
                crop = crop[:, :, :3]

            # Normalize using same percentile normalization as main pipeline
            crop = percentile_normalize(crop)

            # Extract the mask for this crop region
            local_mask = cell_mask[crop_y1:crop_y2, crop_x1:crop_x2]

            # Resize crop and mask to 300x300
            pil_img = Image.fromarray(crop)
            original_size = pil_img.size
            pil_img = pil_img.resize((300, 300), Image.LANCZOS)
            crop_resized = np.array(pil_img)

            # Resize mask to match
            if local_mask.shape[0] > 0 and local_mask.shape[1] > 0:
                mask_pil = Image.fromarray(local_mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((300, 300), Image.NEAREST)
                mask_resized = np.array(mask_pil) > 127

                # Draw solid bright green contour on the image (6px thick)
                crop_with_contour = draw_mask_contour(crop_resized, mask_resized,
                                                       color=(0, 255, 0), dotted=False)
            else:
                crop_with_contour = crop_resized

            # Convert to base64 (JPEG for smaller file sizes)
            pil_img_final = Image.fromarray(crop_with_contour)
            buffer = BytesIO()
            pil_img_final.save(buffer, format='JPEG', quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Use global center from features.json if available, otherwise compute
            if 'center' in feat_dict:
                # center is already in global coordinates
                global_centroid_x = int(feat_dict['center'][0])
                global_centroid_y = int(feat_dict['center'][1])
            else:
                # Backwards compatibility: compute from tile origin + local centroid
                global_centroid_x = tile_x1 + centroid_x
                global_centroid_y = tile_y1 + centroid_y

            # Get global_id if available
            global_id = feat_dict.get('global_id', None)

            samples.append({
                'tile_id': tile_dir.name,
                'det_id': det_id,
                'global_id': global_id,
                'area_px': area_px,
                'area_um2': area_um2,
                'image': img_b64,
                'features': features,
                'solidity': features.get('solidity', 0),
                'circularity': features.get('circularity', 0),
                'global_x': global_centroid_x,
                'global_y': global_centroid_y
            })

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples


def create_mk_hspc_index(output_dir, total_mks, total_hspcs, mk_pages, hspc_pages, slides_summary=None, timestamp=None):
    """
    Create the main index.html page for MK + HSPC batch review.

    Args:
        output_dir: Directory to write index.html
        total_mks: Total number of MK samples
        total_hspcs: Total number of HSPC samples
        mk_pages: Number of MK pages
        hspc_pages: Number of HSPC pages
        slides_summary: Optional string like "16 slides (FGC1, FGC2, ...)"
        timestamp: Segmentation timestamp string
    """
    subtitle_html = f'<p style="color: #888; margin-bottom: 10px;">{slides_summary}</p>' if slides_summary else ''
    timestamp_html = f'<p style="color: #666; font-size: 0.9em;">Segmentation: {timestamp}</p>' if timestamp else ''
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>MK + HSPC Cell Review</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 20px; }}
        .header {{ background: #111; padding: 20px; border: 1px solid #333; margin-bottom: 20px; text-align: center; }}
        h1 {{ font-size: 1.5em; font-weight: normal; margin-bottom: 15px; }}
        .stats {{ display: flex; justify-content: center; gap: 30px; margin: 20px 0; flex-wrap: wrap; }}
        .stat {{ padding: 15px 30px; background: #1a1a1a; border: 1px solid #333; }}
        .stat .number {{ display: block; font-size: 2em; margin-top: 10px; }}
        .section {{ margin: 40px 0; }}
        .section h2 {{ font-size: 1.3em; margin-bottom: 15px; padding: 10px; background: #111; border: 1px solid #333; border-left: 3px solid #555; }}
        .controls {{ text-align: center; margin: 30px 0; }}
        .btn {{ padding: 15px 30px; background: #1a1a1a; border: 1px solid #333; color: #ddd; cursor: pointer; font-family: monospace; font-size: 1.1em; margin: 10px; text-decoration: none; display: inline-block; }}
        .btn:hover {{ background: #222; }}
        .btn-primary {{ border-color: #4a4; color: #4a4; }}
        .btn-export {{ border-color: #44a; color: #44a; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MK + HSPC Cell Review</h1>
        {subtitle_html}
        {timestamp_html}
        <p style="color: #888;">Annotation Interface</p>
        <div class="stats">
            <div class="stat"><span>Total MKs</span><span class="number">{total_mks:,}</span></div>
            <div class="stat"><span>Total HSPCs</span><span class="number">{total_hspcs:,}</span></div>
            <div class="stat"><span>MK Pages</span><span class="number">{mk_pages}</span></div>
            <div class="stat"><span>HSPC Pages</span><span class="number">{hspc_pages}</span></div>
        </div>
    </div>
    <div class="section">
        <h2>Megakaryocytes (MKs)</h2>
        <div class="controls">
            <a href="mk_page1.html" class="btn btn-primary">Review MKs</a>
        </div>
    </div>
    <div class="section">
        <h2>HSPCs</h2>
        <div class="controls">
            <a href="hspc_page1.html" class="btn btn-primary">Review HSPCs</a>
        </div>
    </div>
    <div class="section">
        <h2>Export Annotations</h2>
        <div class="controls">
            <button class="btn btn-export" onclick="exportAnnotations()">Download Annotations JSON</button>
        </div>
    </div>
    <script>
        function exportAnnotations() {{
            const allLabels = {{}};
            const mkLabels = {{ positive: [], negative: [] }};
            const hspcLabels = {{ positive: [], negative: [] }};
            for (let i = 0; i < localStorage.length; i++) {{
                const key = localStorage.key(i);
                if (key.startsWith('mk_labels_page') || key.startsWith('hspc_labels_page')) {{
                    try {{
                        const labels = JSON.parse(localStorage.getItem(key));
                        const cellType = key.startsWith('mk_') ? mkLabels : hspcLabels;
                        for (const [uid, label] of Object.entries(labels)) {{
                            if (label === 1) cellType.positive.push(uid);
                            else if (label === 0) cellType.negative.push(uid);
                        }}
                    }} catch(e) {{ console.error(e); }}
                }}
            }}
            allLabels.mk = mkLabels;
            allLabels.hspc = hspcLabels;
            const blob = new Blob([JSON.stringify(allLabels, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'all_labels_combined.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>'''
    with open(Path(output_dir) / 'index.html', 'w') as f:
        f.write(html)


def generate_mk_hspc_page_html(samples, cell_type, page_num, total_pages, slides_summary=None):
    """
    Generate HTML for a single MK or HSPC annotation page.

    Args:
        samples: List of sample dicts with image data
        cell_type: 'mk' or 'hspc'
        page_num: Current page number (1-indexed)
        total_pages: Total number of pages for this cell type
        slides_summary: Optional string like "16 slides (FGC1, FGC2, ...)" for subtitle

    Returns:
        HTML string for the page
    """
    cell_type_display = "Megakaryocytes (MKs)" if cell_type == "mk" else "HSPCs"

    # Build subtitle HTML
    subtitle_html = ''
    if slides_summary:
        subtitle_html = f'<div class="header-subtitle">{slides_summary}</div>'

    nav_html = '<div class="page-nav">'
    nav_html += '<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{cell_type}_page{page_num-1}.html" class="nav-btn">Previous</a>'
    nav_html += f'<span class="page-info">Page {page_num} of {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{cell_type}_page{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += '</div>'

    cards_html = ""
    for sample in samples:
        slide = sample.get('slide', 'unknown').replace('.', '-')
        global_x = sample.get('global_x', 0)
        global_y = sample.get('global_y', 0)
        # Always use spatial UID format for consistency across all cell types
        # Format: {slide}_{celltype}_{round(x)}_{round(y)}
        uid = f"{slide}_{cell_type}_{int(round(global_x))}_{int(round(global_y))}"
        display_id = f"{cell_type}_{int(round(global_x))}_{int(round(global_y))}"
        # Keep legacy global_id in data attribute for backwards compatibility
        legacy_global_id = sample.get('global_id')
        area_um2 = sample.get('area_um2', 0)
        area_px = sample.get('area_px', 0)
        img_b64 = sample['image']
        # Include legacy_global_id as data attribute for migration support
        legacy_attr = f' data-legacy-id="{legacy_global_id}"' if legacy_global_id is not None else ''
        cards_html += f'''
        <div class="card" id="{uid}" data-label="-1"{legacy_attr}>
            <div class="card-img-container">
                <img src="data:image/jpeg;base64,{img_b64}" alt="{display_id}">
            </div>
            <div class="card-info">
                <div>
                    <div class="card-id">{display_id}</div>
                    <div class="card-area">{area_um2:.1f} um2 | {area_px:.0f} px2</div>
                </div>
                <div class="buttons">
                    <button class="btn btn-yes" onclick="setLabel('{cell_type}', '{uid}', 1)">Yes</button>
                    <button class="btn btn-unsure" onclick="setLabel('{cell_type}', '{uid}', 2)">?</button>
                    <button class="btn btn-no" onclick="setLabel('{cell_type}', '{uid}', 0)">No</button>
                </div>
            </div>
        </div>
'''

    prev_page = page_num - 1
    next_page = page_num + 1

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{cell_type_display} - Page {page_num}/{total_pages}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; }}
        .header {{ background: #111; padding: 12px 20px; display: flex; flex-direction: column; gap: 8px; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 100; }}
        .header-top {{ display: flex; justify-content: space-between; align-items: center; }}
        .header h1 {{ font-size: 1.2em; font-weight: normal; }}
        .header-subtitle {{ font-size: 0.85em; color: #888; margin-top: 2px; }}
        .stats-row {{ display: flex; gap: 20px; font-size: 0.85em; flex-wrap: wrap; }}
        .stats-group {{ display: flex; gap: 8px; align-items: center; }}
        .stats-label {{ color: #888; font-size: 0.9em; }}
        .stat {{ padding: 4px 10px; background: #1a1a1a; border: 1px solid #333; }}
        .stat.positive {{ border-left: 3px solid #4a4; }}
        .stat.negative {{ border-left: 3px solid #a44; }}
        .stat.global {{ background: #0f1a0f; }}
        .page-nav {{ text-align: center; padding: 15px; background: #111; border-bottom: 1px solid #333; }}
        .nav-btn {{ display: inline-block; padding: 8px 16px; margin: 0 10px; background: #1a1a1a; color: #ddd; text-decoration: none; border: 1px solid #333; }}
        .nav-btn:hover {{ background: #222; }}
        .page-info {{ margin: 0 20px; }}
        .content {{ padding: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }}
        .card {{ background: #111; border: 1px solid #333; display: flex; flex-direction: column; }}
        .card-img-container {{ width: 100%; height: 280px; display: flex; align-items: center; justify-content: center; background: #0a0a0a; overflow: hidden; }}
        .card img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
        .card-info {{ padding: 8px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #333; }}
        .card-id {{ font-size: 0.75em; color: #888; }}
        .card-area {{ font-size: 0.8em; }}
        .buttons {{ display: flex; gap: 4px; }}
        .btn {{ padding: 6px 12px; border: 1px solid #333; background: #1a1a1a; color: #ddd; cursor: pointer; font-family: monospace; }}
        .btn:hover {{ background: #222; }}
        .card.labeled-yes {{ border: 3px solid #0f0 !important; background: #131813 !important; }}
        .card.labeled-no {{ border: 3px solid #f00 !important; background: #181111 !important; }}
        .card.labeled-unsure {{ border: 3px solid #fa0 !important; background: #181611 !important; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <h1>{cell_type_display} - Page {page_num}/{total_pages}</h1>
            {subtitle_html}
        </div>
        <div class="stats-row">
            <div class="stats-group">
                <span class="stats-label">This Page:</span>
                <div class="stat">Total: <span id="sample-count">{len(samples)}</span></div>
                <div class="stat positive">Yes: <span id="positive-count">0</span></div>
                <div class="stat negative">No: <span id="negative-count">0</span></div>
            </div>
            <div class="stats-group">
                <span class="stats-label">Global ({total_pages} pages):</span>
                <div class="stat global positive">Yes: <span id="global-positive">0</span></div>
                <div class="stat global negative">No: <span id="global-negative">0</span></div>
            </div>
        </div>
    </div>
    {nav_html}
    <div class="content">
        <div class="grid">{cards_html}</div>
    </div>
    {nav_html}
    <script>
        const STORAGE_KEY = '{cell_type}_labels_page{page_num}';
        const CELL_TYPE = '{cell_type}';
        const TOTAL_PAGES = {total_pages};

        function loadAnnotations() {{
            const stored = localStorage.getItem(STORAGE_KEY);
            if (!stored) return;
            try {{
                const labels = JSON.parse(stored);
                for (const [uid, label] of Object.entries(labels)) {{
                    const card = document.getElementById(uid);
                    if (card && label !== -1) {{
                        card.dataset.label = label;
                        card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                        if (label == 1) card.classList.add('labeled-yes');
                        else if (label == 2) card.classList.add('labeled-unsure');
                        else card.classList.add('labeled-no');
                    }}
                }}
                updateStats();
            }} catch(e) {{ console.error(e); }}
        }}

        function setLabel(cellType, uid, label) {{
            const card = document.getElementById(uid);
            if (!card) return;
            card.dataset.label = label;
            card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            if (label == 1) card.classList.add('labeled-yes');
            else if (label == 2) card.classList.add('labeled-unsure');
            else card.classList.add('labeled-no');
            saveAnnotations();
            updateStats();
        }}

        function saveAnnotations() {{
            const labels = {{}};
            document.querySelectorAll('.card').forEach(card => {{
                const label = parseInt(card.dataset.label);
                if (label !== -1) labels[card.id] = label;
            }});
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
        }}

        function updateStats() {{
            // Local stats (this page)
            let pos = 0, neg = 0;
            document.querySelectorAll('.card').forEach(card => {{
                const label = parseInt(card.dataset.label);
                if (label === 1) pos++;
                else if (label === 0) neg++;
            }});
            document.getElementById('positive-count').textContent = pos;
            document.getElementById('negative-count').textContent = neg;

            // Global stats (all pages)
            let globalPos = 0, globalNeg = 0;
            for (let i = 1; i <= TOTAL_PAGES; i++) {{
                const key = CELL_TYPE + '_labels_page' + i;
                const stored = localStorage.getItem(key);
                if (stored) {{
                    try {{
                        const labels = JSON.parse(stored);
                        for (const label of Object.values(labels)) {{
                            if (label === 1) globalPos++;
                            else if (label === 0) globalNeg++;
                        }}
                    }} catch(e) {{}}
                }}
            }}
            document.getElementById('global-positive').textContent = globalPos;
            document.getElementById('global-negative').textContent = globalNeg;
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' && {page_num} > 1)
                window.location.href = '{cell_type}_page{prev_page}.html';
            else if (e.key === 'ArrowRight' && {page_num} < {total_pages})
                window.location.href = '{cell_type}_page{next_page}.html';
        }});

        loadAnnotations();
    </script>
</body>
</html>'''
    return html


def generate_mk_hspc_pages(samples, cell_type, output_dir, samples_per_page, slides_summary=None):
    """
    Generate separate annotation pages for MK or HSPC samples.

    Args:
        samples: List of sample dicts with image data
        cell_type: 'mk' or 'hspc'
        output_dir: Directory to write HTML files
        samples_per_page: Number of samples per page
        slides_summary: Optional string like "16 slides (FGC1, FGC2, ...)" for subtitle
    """
    if not samples:
        _logger.info(f"  No {cell_type.upper()} samples to export")
        return

    pages = [samples[i:i+samples_per_page] for i in range(0, len(samples), samples_per_page)]
    total_pages = len(pages)

    _logger.info(f"  Generating {total_pages} {cell_type.upper()} pages...")

    for page_num in range(1, total_pages + 1):
        page_samples = pages[page_num - 1]
        html = generate_mk_hspc_page_html(page_samples, cell_type, page_num, total_pages, slides_summary=slides_summary)

        html_path = Path(output_dir) / f"{cell_type}_page{page_num}.html"
        with open(html_path, 'w') as f:
            f.write(html)


def export_mk_hspc_html_from_ram(slide_data, output_base, html_output_dir, samples_per_page=300,
                                  mk_min_area_um=200, mk_max_area_um=2000, timestamp=None):
    """
    Export MK + HSPC HTML annotation pages using slide images already in RAM.

    This is the main entry point for MK/HSPC batch HTML export. It loads sample
    data from the segmentation output directory and generates HTML pages for
    annotation review.

    Args:
        slide_data: dict of {slide_name: {'image': np.array, 'czi_path': path, ...}}
        output_base: Path to segmentation output directory
        html_output_dir: Path to write HTML files
        samples_per_page: Number of samples per HTML page
        mk_min_area_um: Min MK area filter in um^2
        mk_max_area_um: Max MK area filter in um^2
        timestamp: Segmentation timestamp string
    """
    _logger.info(f"\n{'='*70}")
    _logger.info("EXPORTING HTML (using images in RAM)")
    _logger.info(f"{'='*70}")

    html_output_dir = Path(html_output_dir)
    html_output_dir.mkdir(parents=True, exist_ok=True)

    all_mk_samples = []
    all_hspc_samples = []

    PIXEL_SIZE_UM = 0.1725  # Default pixel size

    for slide_name, data in slide_data.items():
        slide_dir = output_base / slide_name
        if not slide_dir.exists():
            continue

        _logger.info(f"  Loading {slide_name}...")

        # Get pixel size from summary
        summary_file = slide_dir / "summary.json"
        pixel_size_um = PIXEL_SIZE_UM
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                ps = summary.get('pixel_size_um')
                if ps:
                    pixel_size_um = ps[0] if isinstance(ps, list) else ps

        slide_image = data['image']

        # Load MK samples (uses largest connected component)
        mk_samples = load_samples_from_ram(
            slide_dir / "mk" / "tiles",
            slide_image, pixel_size_um,
            cell_type='mk'
        )

        # Load HSPC samples (sorted by solidity/confidence)
        hspc_samples = load_samples_from_ram(
            slide_dir / "hspc" / "tiles",
            slide_image, pixel_size_um,
            cell_type='hspc'
        )

        # Add slide name to each sample
        for s in mk_samples:
            s['slide'] = slide_name
        for s in hspc_samples:
            s['slide'] = slide_name

        all_mk_samples.extend(mk_samples)
        all_hspc_samples.extend(hspc_samples)

        _logger.info(f"    {len(mk_samples)} MKs, {len(hspc_samples)} HSPCs")

    # Filter MK by size
    um_to_px_factor = PIXEL_SIZE_UM ** 2
    mk_min_px = int(mk_min_area_um / um_to_px_factor)
    mk_max_px = int(mk_max_area_um / um_to_px_factor)

    mk_before = len(all_mk_samples)
    all_mk_samples = [s for s in all_mk_samples if mk_min_px <= s.get('area_px', 0) <= mk_max_px]
    _logger.info(f"  MK size filter: {mk_before} -> {len(all_mk_samples)}")

    # Sort by area
    all_mk_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)
    all_hspc_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)

    # Build slides summary for subtitle (e.g., "16 slides (FGC1, FGC2, ...)")
    slide_names = sorted(slide_data.keys())
    num_slides = len(slide_names)
    if num_slides > 0:
        # Extract short identifiers (e.g., "FGC1" from "2025_11_18_FGC1")
        short_names = []
        for name in slide_names:
            parts = name.split('_')
            # Take the last part that looks like a group identifier
            short = parts[-1] if parts else name
            short_names.append(short)
        # Show first few names with ellipsis if many
        if len(short_names) > 6:
            preview = ', '.join(short_names[:4]) + ', ...'
        else:
            preview = ', '.join(short_names)
        slides_summary = f"{num_slides} slides ({preview})"
    else:
        slides_summary = None

    # Generate pages
    generate_mk_hspc_pages(all_mk_samples, "mk", html_output_dir, samples_per_page, slides_summary=slides_summary)
    generate_mk_hspc_pages(all_hspc_samples, "hspc", html_output_dir, samples_per_page, slides_summary=slides_summary)

    # Create index
    mk_pages = (len(all_mk_samples) + samples_per_page - 1) // samples_per_page if all_mk_samples else 0
    hspc_pages = (len(all_hspc_samples) + samples_per_page - 1) // samples_per_page if all_hspc_samples else 0
    create_mk_hspc_index(html_output_dir, len(all_mk_samples), len(all_hspc_samples), mk_pages, hspc_pages, slides_summary=slides_summary, timestamp=timestamp)

    _logger.info(f"\n  HTML export complete: {html_output_dir}")
    _logger.info(f"  Total: {len(all_mk_samples)} MKs, {len(all_hspc_samples)} HSPCs")


# Backward compatibility aliases (original function names)
create_export_index = create_mk_hspc_index
generate_export_page_html = generate_mk_hspc_page_html
generate_export_pages = generate_mk_hspc_pages
export_html_from_ram = export_mk_hspc_html_from_ram
