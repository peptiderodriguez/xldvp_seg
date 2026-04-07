"""CSS style generators for HTML annotation pages.

Extracted from ``html_export.py``. Contains:

**get_css():** Unified CSS for standard annotation pages (cell, NMJ, MK, etc.)
**get_vessel_css():** Enhanced CSS for vessel annotation with batch selection,
    filtering, and RF training export support.
"""


def get_css():
    """Get the unified CSS styles."""
    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: monospace; background: #0a0a0a; color: #ddd; }

        .header {
            background: #111;
            padding: 12px 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            border-bottom: 1px solid #333;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .header h1 {
            font-size: 1.2em;
            font-weight: normal;
        }

        .header-subtitle {
            font-size: 0.85em;
            color: #888;
            margin-top: 2px;
        }

        .nav-buttons {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .nav-btn {
            padding: 8px 15px;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #ddd;
            text-decoration: none;
            cursor: pointer;
            font-family: monospace;
        }

        .nav-btn:hover {
            background: #222;
        }

        .page-info {
            padding: 8px 15px;
            color: #888;
        }

        .stats-row {
            display: flex;
            gap: 20px;
            font-size: 0.85em;
            flex-wrap: wrap;
            align-items: center;
        }

        .stats-group {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .stats-label {
            color: #888;
            font-size: 0.9em;
        }

        .stat {
            padding: 4px 10px;
            background: #1a1a1a;
            border: 1px solid #333;
        }

        .stat.positive {
            border-left: 3px solid #4a4;
        }

        .stat.negative {
            border-left: 3px solid #a44;
        }

        .channel-legend {
            margin-left: auto;
            padding: 4px 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
        }

        .channel-legend span {
            margin: 0 6px;
            font-weight: bold;
        }

        .ch-red { color: #ff6666; }
        .ch-green { color: #66ff66; }
        .ch-blue { color: #6666ff; }

        .content {
            padding: 15px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
        }

        .card {
            background: #111;
            border: 2px solid #333;
            overflow: hidden;
            transition: border-color 0.2s;
        }

        .card.selected {
            box-shadow: 0 0 0 3px #fff;
        }

        .card.labeled-yes {
            border-color: #4a4 !important;
            background: #0f130f !important;
        }

        .card.labeled-no {
            border-color: #a44 !important;
            background: #130f0f !important;
        }

        .card.labeled-unsure {
            border-color: #aa4 !important;
            background: #13130f !important;
        }

        .card-img-container {
            width: 100%;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
        }

        .card img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .card-info {
            padding: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #333;
            gap: 10px;
        }

        .card-meta {
            flex: 1;
            min-width: 0;
        }

        .card-id {
            font-size: 0.75em;
            color: #888;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .card-stats {
            font-size: 0.8em;
            color: #aaa;
            margin-top: 3px;
        }

        .buttons {
            display: flex;
            gap: 5px;
            flex-shrink: 0;
        }

        .btn {
            padding: 6px 12px;
            border: 1px solid #333;
            background: #1a1a1a;
            color: #ddd;
            cursor: pointer;
            font-family: monospace;
            font-size: 0.85em;
        }

        .btn:hover {
            background: #222;
        }

        .btn-yes {
            border-color: #4a4;
            color: #4a4;
        }

        .btn-no {
            border-color: #a44;
            color: #a44;
        }

        .btn-unsure {
            border-color: #aa4;
            color: #aa4;
        }

        .btn-export {
            border-color: #44a;
            color: #44a;
        }

        .btn-danger {
            border-color: #a44;
            color: #a44;
        }

        .btn-danger:hover {
            background: #311;
        }

        .keyboard-hint {
            text-align: center;
            padding: 15px;
            color: #555;
            font-size: 0.85em;
            border-top: 1px solid #222;
        }

        .footer {
            background: #111;
            padding: 15px;
            border-top: 1px solid #333;
            display: flex;
            justify-content: center;
            gap: 10px;
        }
    """


def get_vessel_css():
    """Get enhanced CSS styles for vessel annotation interface."""
    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: monospace; background: #0a0a0a; color: #ddd; }

        .header {
            background: #111;
            padding: 12px 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            border-bottom: 1px solid #333;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .header h1 {
            font-size: 1.2em;
            font-weight: normal;
        }

        .header-subtitle {
            font-size: 0.85em;
            color: #888;
            margin-top: 2px;
        }

        .nav-buttons {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .nav-btn {
            padding: 8px 15px;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #ddd;
            text-decoration: none;
            cursor: pointer;
            font-family: monospace;
        }

        .nav-btn:hover {
            background: #222;
        }

        .page-info {
            padding: 8px 15px;
            color: #888;
        }

        /* Filter panel */
        .filter-panel {
            background: #0d0d0d;
            padding: 12px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }

        .filter-group {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .filter-group label {
            color: #888;
            font-size: 0.85em;
        }

        .filter-group input[type="number"],
        .filter-group select {
            padding: 5px 8px;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #ddd;
            font-family: monospace;
            width: 80px;
        }

        .filter-group input[type="checkbox"] {
            width: 16px;
            height: 16px;
        }

        .filter-btn {
            padding: 6px 12px;
            background: #1a1a1a;
            border: 1px solid #44a;
            color: #44a;
            cursor: pointer;
            font-family: monospace;
        }

        .filter-btn:hover {
            background: #0f0f13;
        }

        /* Stats row */
        .stats-row {
            display: flex;
            gap: 20px;
            font-size: 0.85em;
            flex-wrap: wrap;
            align-items: center;
            padding: 8px 0;
        }

        .stats-group {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .stats-label {
            color: #888;
            font-size: 0.9em;
        }

        .stat {
            padding: 4px 10px;
            background: #1a1a1a;
            border: 1px solid #333;
        }

        .stat.positive {
            border-left: 3px solid #4a4;
        }

        .stat.negative {
            border-left: 3px solid #a44;
        }

        .stat.remaining {
            border-left: 3px solid #aa4;
        }

        /* Batch selection toolbar */
        .batch-toolbar {
            background: #111;
            padding: 10px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .batch-toolbar.hidden {
            display: none;
        }

        .batch-btn {
            padding: 6px 12px;
            border: 1px solid #333;
            background: #1a1a1a;
            color: #ddd;
            cursor: pointer;
            font-family: monospace;
            font-size: 0.85em;
        }

        .batch-btn:hover {
            background: #222;
        }

        .batch-btn-yes {
            border-color: #4a4;
            color: #4a4;
        }

        .batch-btn-no {
            border-color: #a44;
            color: #a44;
        }

        .batch-count {
            color: #888;
        }

        /* Content area */
        .content {
            padding: 15px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
        }

        /* Card styles */
        .card {
            background: #111;
            border: 2px solid #333;
            overflow: hidden;
            transition: border-color 0.2s, transform 0.1s;
            position: relative;
        }

        .card.selected {
            box-shadow: 0 0 0 3px #fff;
        }

        .card.batch-selected {
            box-shadow: 0 0 0 3px #44a;
        }

        .card.labeled-yes {
            border-color: #4a4 !important;
            background: #0f130f !important;
        }

        .card.labeled-no {
            border-color: #a44 !important;
            background: #130f0f !important;
        }

        .card.labeled-unsure {
            border-color: #aa4 !important;
            background: #13130f !important;
        }

        .card.hidden {
            display: none;
        }

        .card-checkbox {
            position: absolute;
            top: 8px;
            left: 8px;
            width: 20px;
            height: 20px;
            z-index: 10;
        }

        .card-img-container {
            width: 100%;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
            cursor: pointer;
        }

        .card img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Side-by-side image layout for raw + contours */
        .card-img-sidebyside {
            display: flex;
            flex-direction: row;
            gap: 2px;
        }

        .img-half {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
        }

        .img-half img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .img-label {
            position: absolute;
            bottom: 4px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.7);
            color: #aaa;
            font-size: 0.65em;
            padding: 2px 6px;
            border-radius: 3px;
        }

        .card-info {
            padding: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #333;
            gap: 10px;
        }

        .card-meta {
            flex: 1;
            min-width: 0;
        }

        .card-id {
            font-size: 0.75em;
            color: #888;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .card-stats {
            font-size: 0.8em;
            color: #aaa;
            margin-top: 3px;
        }

        .card-features {
            font-size: 0.7em;
            color: #666;
            margin-top: 2px;
        }

        .buttons {
            display: flex;
            gap: 5px;
            flex-shrink: 0;
        }

        .btn {
            padding: 6px 12px;
            border: 1px solid #333;
            background: #1a1a1a;
            color: #ddd;
            cursor: pointer;
            font-family: monospace;
            font-size: 0.85em;
        }

        .btn:hover {
            background: #222;
        }

        .btn-yes {
            border-color: #4a4;
            color: #4a4;
        }

        .btn-no {
            border-color: #a44;
            color: #a44;
        }

        .btn-unsure {
            border-color: #aa4;
            color: #aa4;
        }

        .btn-export {
            border-color: #44a;
            color: #44a;
        }

        .btn-danger {
            border-color: #a44;
            color: #a44;
        }

        .btn-danger:hover {
            background: #311;
        }

        /* Export panel */
        .export-panel {
            background: #111;
            padding: 15px 20px;
            border-top: 1px solid #333;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            justify-content: center;
        }

        .keyboard-hint {
            text-align: center;
            padding: 15px;
            color: #555;
            font-size: 0.85em;
            border-top: 1px solid #222;
        }

        .footer {
            background: #111;
            padding: 15px;
            border-top: 1px solid #333;
            display: flex;
            justify-content: center;
            gap: 10px;
        }

        /* Modal for export preview */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .modal.show {
            display: flex;
        }

        .modal-content {
            background: #111;
            border: 1px solid #333;
            padding: 20px;
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .modal-close {
            background: none;
            border: none;
            color: #888;
            font-size: 1.5em;
            cursor: pointer;
        }

        .modal-close:hover {
            color: #fff;
        }

        pre {
            background: #0a0a0a;
            padding: 15px;
            overflow: auto;
            max-height: 400px;
            font-size: 0.85em;
        }
    """
