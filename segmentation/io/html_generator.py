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
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
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
        nav_parts.append('<a href="index.html" class="nav-btn">Home</a>')

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
        const PAGE_NUM = typeof PAGE_NUM !== 'undefined' ? PAGE_NUM : 1;
        const STORAGE_KEY = CELL_TYPE + '_labels_page' + PAGE_NUM;
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
