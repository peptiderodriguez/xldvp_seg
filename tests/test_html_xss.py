"""XSS regression tests for Phase 2 hardening.

Canary UID payload: breaking-out-of-JS-string + breaking-out-of-<script>.
Ensures the nested _esc(_js_esc(...)) escape + safe_json + textContent fixes
hold against adversarial input.
"""

import pytest

from xldvp_seg.io.html_utils import _esc, _esc_js_in_attr, _js_esc


class TestJsEscHardening:
    def test_escapes_closing_script_tag(self):
        """</script> inside a JS string literal must be broken so it can't
        terminate the enclosing <script> block."""
        out = _js_esc("</script>alert(1)")
        assert "</script>" not in out
        # The escaped form keeps the characters but separates them.
        assert "<\\/" in out

    def test_escapes_closing_comment_sequence(self):
        """*/ inside a JS comment must be broken."""
        out = _js_esc("*/;fetch('http://evil');/*")
        assert "*/" not in out

    def test_quote_escaped(self):
        """A single quote inside _js_esc output is JSON-safe."""
        out = _js_esc("hello' onclick=alert(1) '")
        # json.dumps wraps strings in " and escapes internal " only; single
        # quotes pass through unchanged (JSON allows them). The outer context
        # should use &quot; delimiters, which _esc(_js_esc(...)) gives.
        assert "'" in out  # single quote is allowed in JSON strings

    def test_double_quote_escaped(self):
        """A double quote inside _js_esc output is escaped."""
        out = _js_esc('say "hi"')
        assert '\\"' in out


class TestNestedEscape:
    def test_payload_in_onclick_attribute(self):
        """The nested _esc(_js_esc(uid)) pattern used in onclick attributes
        must produce a value that is inert inside &quot;...&quot; delimiters."""
        payload = "a\"';alert(1);//"
        rendered = _esc(_js_esc(payload))
        # Must not contain un-escaped &quot; breakout
        assert "&quot;" in rendered or '"' not in rendered.replace("&quot;", "")
        # Must not contain raw <script> / </script>
        assert "<script" not in rendered.lower()
        assert "</script" not in rendered.lower()

    def test_esc_js_in_attr_helper_matches_nested(self):
        """Phase D.2: _esc_js_in_attr must produce the same output as the
        manual _esc(_js_esc(x)) composition it replaces."""
        for payload in ["a\"';alert(1);//", "</script>", "*/", "normal_uid"]:
            assert _esc_js_in_attr(payload) == _esc(_js_esc(payload))


class TestSafeJsonInScript:
    def test_safe_json_escapes_script_close(self):
        from xldvp_seg.visualization.encoding import safe_json

        payload = {"key</script><script>alert(1)</script>end": 1.0}
        out = safe_json(payload)
        # safe_json does `</` → `<\/`; bare </script> must not remain.
        assert "</script>" not in out
        assert "<\\/script>" in out


class TestRegionViewerNoInnerHTML:
    """Verify the region_viewer JS source no longer concatenates layer names
    via innerHTML — it must use textContent + createElement instead.
    """

    def test_region_viewer_uses_textContent(self):
        from xldvp_seg.visualization import region_viewer

        source = region_viewer.__file__
        with open(source) as f:
            text = f.read()
        assert "textContent = l.name" in text
        # The old pattern `l.name + '</span>'` must be gone.
        assert "'<span>' +" not in text
        assert "'</span>' + tag" not in text


class TestOnclickSitesUseNestedEscape:
    """Grep the HTML-generation modules to verify every onclick="setLabel(...)"
    uses the &quot;{uid_js}&quot; pattern, not the old '{uid}'."""

    @pytest.mark.parametrize(
        "module_file",
        [
            "xldvp_seg/io/html_export.py",
            "xldvp_seg/io/html_generator.py",
            "xldvp_seg/io/html_batch_export.py",
        ],
    )
    def test_no_single_quote_onclick(self, module_file):
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        text = (repo / module_file).read_text()
        # Old vulnerable pattern: onclick="setLabel('{uid}', N)"
        assert (
            "onclick=\"setLabel('{uid}'" not in text
        ), f"{module_file} still contains pre-Phase-2.2 onclick pattern"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
