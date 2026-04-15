#!/usr/bin/env python3
"""Export a Claude Code session JSONL as a clean transcript (markdown or HTML).

Renders user messages, assistant text, AskUserQuestion prompts + selected
answers, and condensed tool calls. Designed for presentations / showcase
material where the full raw JSONL would be unreadable.

Two HTML modes:

  curate   Keep/skip checkboxes, "Export Curated" button to drop unwanted
           steps and download a cleaned HTML. Use this first to review.

  present  No curation UI. Adds "Download PNGs" (one file per card) and
           "Combined PNG" buttons for dropping into slide decks. Run
           against your curated HTML when you're ready to present.

Basic usage:

    # Markdown + curation HTML (default)
    python scripts/export_transcript.py session.jsonl --output-dir out/

    # Presentation HTML with PNG export (after curation)
    python scripts/export_transcript.py session.jsonl \\
        --mode present --format html --output-dir out/

Finding your session file:

    ls ~/.claude/projects/*/<session-id>.jsonl

Keyboard shortcuts in the HTML viewer:

    →/Space/j       next step       ←/k             prev step
    Home            restart         End             show all
    a               show all        c               toggle keep (curate mode)
    s               show checks     (buttons in the toolbar work too)
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

ANSWER_RE = re.compile(r'"([^"]+?)"="([^"]+?)"')


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_answer_string(content: str) -> list[tuple[str, str]]:
    """Extract (question, answer) pairs from an AskUserQuestion tool_result."""
    return ANSWER_RE.findall(content)


def text_blocks(content: Any) -> list[str]:
    """Return all text blocks from a message content (string or block list)."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return out


def tool_uses(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def tool_results(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


# ---------------------------------------------------------------------------
# Filtering — drop noise that hurts a presentation transcript
# ---------------------------------------------------------------------------

SLASH_CMD_RE = re.compile(r"<command-name>([^<]+)</command-name>", re.I)
SYS_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
COMMAND_TAGS_RE = re.compile(
    r"<command-(?:message|name|args)>[^<]*</command-(?:message|name|args)>"
)


def looks_like_slash_command_expansion(text: str) -> bool:
    """True if text looks like the long body of a slash command expansion."""
    if not text:
        return False
    starts = ("You are ", "You're ", "Act as ", "# ", "## ")
    return len(text) > 1500 and text.lstrip().startswith(starts)


def clean_user_text(text: str) -> str:
    text = SYS_REMINDER_RE.sub("", text)
    text = COMMAND_TAGS_RE.sub("", text)
    return text.strip()


def _summarize_input(name: str, inp: dict) -> str:
    if name in {"Read", "Edit", "Write"}:
        return inp.get("file_path", "")
    if name == "Glob":
        return inp.get("pattern", "")
    if name == "Grep":
        return f"pattern={inp.get('pattern', '')!r} path={inp.get('path', '')}"
    return ""


def _find_tool_output(records: list[dict], tool_use_id: str) -> str:
    if not tool_use_id:
        return ""
    for rec in records:
        if rec.get("type") != "user":
            continue
        msg = rec.get("message") or {}
        for tr in tool_results(msg.get("content")):
            if tr.get("tool_use_id") == tool_use_id:
                c = tr.get("content", "")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return "\n".join(b.get("text", "") for b in c if isinstance(b, dict))
                return ""
    return ""


def _snippet(text: str, max_lines: int) -> tuple[str, bool]:
    if max_lines <= 0:
        return "", False
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines), False
    return "\n".join(lines[:max_lines]), True


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_markdown(records: list[dict], bash_output_lines: int) -> str:
    lines: list[str] = ["# Session Transcript", ""]
    pending_questions: dict[str, dict] = {}

    for rec in records:
        rtype = rec.get("type")
        msg = rec.get("message") or {}
        content = msg.get("content")

        if rtype == "user":
            for tr in tool_results(content):
                tid = tr.get("tool_use_id", "")
                tcontent = tr.get("content", "")
                if (
                    tid in pending_questions
                    and isinstance(tcontent, str)
                    and "has answered" in tcontent
                ):
                    q_input = pending_questions.pop(tid)
                    answers = dict(parse_answer_string(tcontent))
                    lines.append("**User answers:**")
                    for q in q_input.get("questions", []):
                        qtext = q.get("question", "")
                        ans = answers.get(qtext, "_(no answer captured)_")
                        lines.append(f"- _{qtext}_ → **{ans}**")
                    lines.append("")

            for txt in text_blocks(content):
                txt = clean_user_text(txt)
                if not txt:
                    continue
                slash = SLASH_CMD_RE.search(txt)
                if slash:
                    lines.append(f"## /{slash.group(1).strip().lstrip('/')} (user invoked)")
                    lines.append("")
                    continue
                if looks_like_slash_command_expansion(txt):
                    lines.append("> _(slash-command system prompt body — omitted)_")
                    lines.append("")
                    continue
                lines.append("## User")
                lines.append("")
                lines.append(txt)
                lines.append("")

        elif rtype == "assistant":
            texts = [t for t in text_blocks(content) if t.strip()]
            tools = tool_uses(content)
            if texts or tools:
                lines.append("## Assistant")
                lines.append("")
            for t in texts:
                lines.append(t.rstrip())
                lines.append("")
            for tu in tools:
                name = tu.get("name", "")
                inp = tu.get("input", {})
                if name == "AskUserQuestion":
                    pending_questions[tu.get("id", "")] = inp
                    for q in inp.get("questions", []):
                        lines.append(f"### ❓ {q.get('question', '')}")
                        for opt in q.get("options", []):
                            label = opt.get("label", "")
                            desc = opt.get("description", "")
                            lines.append(f"- **{label}** — {desc}")
                        lines.append("")
                elif name == "Bash":
                    cmd = inp.get("command", "")
                    desc = inp.get("description", "")
                    lines.append(f"_Tool: Bash — {desc}_" if desc else "_Tool: Bash_")
                    lines.append("```bash")
                    lines.append(cmd.strip())
                    lines.append("```")
                    out = _find_tool_output(records, tu.get("id", ""))
                    if out:
                        snipped, truncated = _snippet(out, bash_output_lines)
                        if snipped.strip():
                            lines.append("```")
                            lines.append(snipped)
                            lines.append("```")
                            if truncated:
                                lines.append("_(output truncated)_")
                    lines.append("")
                elif name in {"Read", "Edit", "Write", "Glob", "Grep"}:
                    summary = _summarize_input(name, inp)
                    lines.append(f"_Tool: {name} — {summary}_")
                    lines.append("")
                else:
                    lines.append(f"_Tool: {name}_")
                    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# HTML renderer (shared head/body, mode-specific toolbar + script)
# ---------------------------------------------------------------------------

HTML_CSS = """<style>
  :root { --user: #1a7a3e; --assistant: #2d4a8a; --tool: #9b6a00; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 960px; margin: 0 auto; padding: 5rem 1.25rem 6rem;
         color: #1a1a1a; line-height: 1.55; background: #fafafa; }
  h1 { border-bottom: 2px solid #ccc; padding-bottom: .3rem; margin-top: 0; }
  h2 { margin-top: 2.5rem; color: var(--assistant); }
  h2.user { color: var(--user); }
  .role { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;
          color: #888; margin-bottom: .3rem; }
  .msg { background: #fff; padding: 1rem 1.25rem; border-radius: 6px;
         border: 1px solid #e0e0e0; margin-bottom: 1rem;
         box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  .msg.user { border-left: 4px solid var(--user); }
  .msg.assistant { border-left: 4px solid var(--assistant); }
  .msg.tool { border-left: 4px solid var(--tool); background: #fcf9f3; }
  .msg.answers { border-left: 4px solid var(--user); background: #f3faf5; }
  pre { background: #1e1e1e; color: #e4e4e4; padding: .75rem 1rem;
        border-radius: 4px; overflow-x: auto; font-size: .85rem; }
  code { background: #eee; padding: 1px 5px; border-radius: 3px; font-size: .9em; }
  pre code { background: transparent; padding: 0; }
  .question-card { background: #f5f7fb; border: 1px solid #d4dbe8;
                   border-radius: 8px; padding: 1rem; margin: .5rem 0; }
  .question-card .q { font-weight: 600; margin-bottom: .6rem; color: var(--assistant); }
  .options { display: grid; gap: .5rem; }
  .option { background: #fff; border: 1px solid #d0d7e2; padding: .55rem .8rem;
            border-radius: 5px; cursor: default; }
  .option .label { font-weight: 600; }
  .option .desc { color: #555; font-size: .87rem; margin-top: .15rem; }
  .option.selected { border: 2px solid var(--user); background: #eaf6ee; }
  .option.selected .label::before { content: "\\2713 "; color: var(--user); }
  .tool-name { font-size: .8rem; color: var(--tool); text-transform: uppercase;
               letter-spacing: .08em; margin-bottom: .4rem; }
  .truncated { color: #888; font-style: italic; font-size: .85rem; }
  .skipped { color: #aaa; font-style: italic; }

  /* Presentation mode: reveal steps progressively */
  .step { transition: opacity .25s ease; position: relative; }
  body.present .step { display: none; }
  body.present .step.shown { display: block; animation: fadein .35s ease; }
  body.present .step.current { box-shadow: 0 0 0 3px rgba(45,74,138,.25); }
  @keyframes fadein { from { opacity: 0; transform: translateY(8px); }
                       to   { opacity: 1; transform: translateY(0); } }

  /* Curation checkbox (curate mode only) */
  .step-check { position: absolute; top: .5rem; right: .5rem; z-index: 5;
                background: rgba(255,255,255,.85); border-radius: 4px;
                padding: 2px 6px; font-size: .75rem; color: #555;
                border: 1px solid #ddd; user-select: none; cursor: pointer;
                display: flex; align-items: center; gap: .35rem; }
  .step-check input { margin: 0; cursor: pointer; accent-color: var(--user); }
  .step.unchecked { opacity: .35; background: #f4f4f4; }
  .step.unchecked .step-check { background: #fff5f5; color: #b33;
                                 border-color: #f4caca; }
  body.curate-hide .step.unchecked { display: none !important; }
  body.present .step-check { display: none; }
  body.present.show-checks .step-check { display: flex; }

  .toolbar { position: fixed; top: 0; left: 0; right: 0; height: 44px;
             background: #1e2937; color: #eee; display: flex; align-items: center;
             padding: 0 1rem; gap: 1rem; z-index: 100;
             box-shadow: 0 2px 6px rgba(0,0,0,.15); font-size: .9rem; }
  .toolbar button { background: #364456; color: #eee; border: none;
                    padding: .35rem .75rem; border-radius: 4px; cursor: pointer;
                    font-size: .85rem; }
  .toolbar button:hover { background: #4a5a70; }
  .toolbar button:disabled { opacity: .5; cursor: wait; }
  .toolbar .counter { margin-left: auto; font-variant-numeric: tabular-nums;
                      color: #aac; }
  .toolbar .hint { color: #88a; font-size: .8rem; }
  .progress { position: fixed; top: 44px; left: 0; height: 3px;
              background: #2d4a8a; z-index: 99; transition: width .25s ease; }
</style>"""

CURATE_TOOLBAR = """<div class="toolbar">
  <button id="prev">\u2190 Prev</button>
  <button id="next">Next \u2192</button>
  <button id="toggle">Show All</button>
  <button id="restart">Restart</button>
  <button id="hide-unchecked" title="Toggle visibility of unchecked steps">Hide Unchecked</button>
  <button id="export">Export Curated</button>
  <button id="check-all" title="Mark all as keep">Check All</button>
  <button id="uncheck-all" title="Mark all as skip">Uncheck All</button>
  <span class="hint">Arrows advance \u00b7 click \u2611 to keep/skip</span>
  <span class="counter" id="counter">0 / 0</span>
</div>"""

CURATE_SCRIPT = r"""<script>
(function(){
  const steps = Array.from(document.querySelectorAll('.step'));
  const counter = document.getElementById('counter');
  const progress = document.getElementById('progress');
  const body = document.body;
  const STORE_KEY = 'transcript-curation::' + location.pathname;

  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); }
  catch (e) { saved = {}; }

  steps.forEach((el, i) => {
    el.dataset.stepIdx = i;
    const wrap = document.createElement('label');
    wrap.className = 'step-check';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = saved[i] !== false;
    if (!cb.checked) el.classList.add('unchecked');
    cb.addEventListener('change', () => {
      el.classList.toggle('unchecked', !cb.checked);
      saved[i] = cb.checked;
      try { localStorage.setItem(STORE_KEY, JSON.stringify(saved)); } catch(e) {}
      renderCounter();
    });
    const txt = document.createElement('span');
    txt.textContent = 'keep';
    wrap.appendChild(cb);
    wrap.appendChild(txt);
    el.appendChild(wrap);
  });

  let idx = 0;
  function visibleSteps() {
    if (body.classList.contains('curate-hide'))
      return steps.filter(el => !el.classList.contains('unchecked'));
    return steps;
  }
  function renderCounter() {
    const vs = visibleSteps();
    const kept = steps.filter(el => !el.classList.contains('unchecked')).length;
    counter.textContent = idx + ' / ' + vs.length + ' (kept ' + kept + '/' + steps.length + ')';
    progress.style.width = vs.length ? (100 * idx / vs.length) + '%' : '0%';
  }
  function render() {
    const vs = visibleSteps();
    steps.forEach(el => { el.classList.remove('shown', 'current'); });
    vs.slice(0, idx).forEach((el, i) => {
      el.classList.add('shown');
      if (i === idx - 1) el.classList.add('current');
    });
    renderCounter();
    if (idx > 0 && body.classList.contains('present')) {
      const el = vs[idx - 1];
      if (el) el.scrollIntoView({behavior:'smooth', block:'center'});
    }
  }
  function next() { const vs = visibleSteps(); if (idx < vs.length) { idx++; render(); } }
  function prev() { if (idx > 0) { idx--; render(); } }
  function showAll() {
    body.classList.remove('present');
    body.classList.remove('show-checks');
    steps.forEach(el => el.classList.add('shown'));
    renderCounter();
  }
  function restart() { body.classList.add('present'); idx = 0; render(); }

  document.getElementById('next').onclick = next;
  document.getElementById('prev').onclick = prev;
  document.getElementById('toggle').onclick = () => body.classList.contains('present') ? showAll() : restart();
  document.getElementById('restart').onclick = restart;
  document.getElementById('hide-unchecked').onclick = (e) => {
    body.classList.toggle('curate-hide');
    e.target.textContent = body.classList.contains('curate-hide') ? 'Show Unchecked' : 'Hide Unchecked';
    idx = 0; render();
  };
  document.getElementById('check-all').onclick = () => {
    document.querySelectorAll('.step-check input').forEach(cb => {
      cb.checked = true; cb.dispatchEvent(new Event('change'));
    });
  };
  document.getElementById('uncheck-all').onclick = () => {
    document.querySelectorAll('.step-check input').forEach(cb => {
      cb.checked = false; cb.dispatchEvent(new Event('change'));
    });
  };
  document.getElementById('export').onclick = () => {
    const clone = document.documentElement.cloneNode(true);
    clone.querySelectorAll('.step.unchecked').forEach(el => el.remove());
    clone.querySelectorAll('.step-check').forEach(el => el.remove());
    const blob = new Blob(
      ['<!doctype html>\n<html lang="en">' + clone.innerHTML + '</html>'],
      {type: 'text/html'}
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const stamp = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
    a.href = url;
    a.download = (document.title || 'transcript') + '_curated_' + stamp + '.html';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
  };

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown' || e.key === 'j') {
      e.preventDefault(); next();
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp' || e.key === 'k') {
      e.preventDefault(); prev();
    } else if (e.key === 'Home') { e.preventDefault(); restart(); }
    else if (e.key === 'End') { e.preventDefault(); idx = visibleSteps().length; render(); }
    else if (e.key === 'a') { e.preventDefault(); showAll(); }
    else if (e.key === 'c') {
      e.preventDefault();
      const vs = visibleSteps();
      const cur = vs[idx - 1];
      if (cur) {
        const cb = cur.querySelector('.step-check input');
        if (cb) { cb.checked = !cb.checked; cb.dispatchEvent(new Event('change')); }
      }
    } else if (e.key === 's') {
      e.preventDefault(); body.classList.toggle('show-checks');
    }
  });

  body.classList.add('show-checks');
  render();
})();
</script>"""

PRESENT_TOOLBAR = """<div class="toolbar">
  <button id="prev">\u2190 Prev</button>
  <button id="next">Next \u2192</button>
  <button id="toggle">Show All</button>
  <button id="restart">Restart</button>
  <button id="pngbtn" title="Download one PNG per card">Download PNGs</button>
  <button id="combinedbtn" title="Download a single PNG of all cards stacked">Combined PNG</button>
  <span class="hint">Arrow keys / Space to advance</span>
  <span class="counter" id="counter">0 / 0</span>
</div>"""

PRESENT_SCRIPT = r"""<script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js" crossorigin="anonymous"></script>
<script>
(function(){
  const steps = Array.from(document.querySelectorAll('.step'));
  const counter = document.getElementById('counter');
  const progress = document.getElementById('progress');
  const body = document.body;
  let idx = 0;

  function render() {
    steps.forEach((el, i) => {
      el.classList.toggle('shown', i < idx);
      el.classList.toggle('current', i === idx - 1);
    });
    counter.textContent = idx + ' / ' + steps.length;
    progress.style.width = steps.length ? (100 * idx / steps.length) + '%' : '0%';
    if (idx > 0 && body.classList.contains('present')) {
      const el = steps[idx - 1];
      if (el) el.scrollIntoView({behavior:'smooth', block:'center'});
    }
  }
  function next() { if (idx < steps.length) { idx++; render(); } }
  function prev() { if (idx > 0) { idx--; render(); } }
  function showAll() {
    body.classList.remove('present');
    steps.forEach(el => el.classList.add('shown'));
    counter.textContent = steps.length + ' / ' + steps.length;
    progress.style.width = '100%';
  }
  function restart() { body.classList.add('present'); idx = 0; render(); }

  document.getElementById('next').onclick = next;
  document.getElementById('prev').onclick = prev;
  document.getElementById('restart').onclick = restart;
  document.getElementById('toggle').onclick = () => body.classList.contains('present') ? showAll() : restart();

  function kindOf(el) {
    return el.classList.contains('answers') ? 'answer'
         : el.classList.contains('user') ? 'user'
         : el.classList.contains('assistant') ? 'assistant'
         : el.classList.contains('tool') ? 'tool' : 'step';
  }
  async function withAllShown(fn) {
    const wasPresent = body.classList.contains('present');
    if (wasPresent) showAll();
    try { return await fn(); }
    finally { if (wasPresent) restart(); }
  }

  document.getElementById('pngbtn').addEventListener('click', async () => {
    if (typeof html2canvas === 'undefined') {
      alert('html2canvas failed to load. You need internet (loads from CDN).');
      return;
    }
    const btn = document.getElementById('pngbtn');
    btn.disabled = true;
    await withAllShown(async () => {
      for (let i = 0; i < steps.length; i++) {
        const el = steps[i];
        btn.textContent = 'Rendering ' + (i + 1) + ' / ' + steps.length;
        const canvas = await html2canvas(el, {backgroundColor:'#fafafa', scale:2, logging:false, useCORS:true});
        const blob = await new Promise(res => canvas.toBlob(res, 'image/png'));
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const num = String(i + 1).padStart(2, '0');
        a.href = url;
        a.download = 'step_' + num + '_' + kindOf(el) + '.png';
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url);
        await new Promise(r => setTimeout(r, 150));
      }
    });
    btn.textContent = 'Download PNGs';
    btn.disabled = false;
  });

  document.getElementById('combinedbtn').addEventListener('click', async () => {
    if (typeof html2canvas === 'undefined') {
      alert('html2canvas failed to load. You need internet (loads from CDN).');
      return;
    }
    const btn = document.getElementById('combinedbtn');
    btn.disabled = true;
    btn.textContent = 'Rendering combined\u2026';
    await withAllShown(async () => {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'background:#fafafa;padding:1.5rem;max-width:960px;'
                         + 'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;';
      steps.forEach(el => wrap.appendChild(el.cloneNode(true)));
      document.body.appendChild(wrap);
      const canvas = await html2canvas(wrap, {backgroundColor:'#fafafa', scale:2, logging:false, useCORS:true, windowWidth:1000});
      wrap.remove();
      const blob = await new Promise(res => canvas.toBlob(res, 'image/png'));
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'transcript_combined.png';
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    });
    btn.textContent = 'Combined PNG';
    btn.disabled = false;
  });

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown' || e.key === 'j') {
      e.preventDefault(); next();
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp' || e.key === 'k') {
      e.preventDefault(); prev();
    } else if (e.key === 'Home') { e.preventDefault(); restart(); }
    else if (e.key === 'End') { e.preventDefault(); idx = steps.length; render(); }
    else if (e.key === 'a') { e.preventDefault(); showAll(); }
  });

  render();
})();
</script>"""


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _head(title: str) -> str:
    return (
        '<!doctype html>\n<html lang="en"><head>\n'
        f'<meta charset="utf-8"><title>{_esc(title)}</title>\n'
        f'{HTML_CSS}\n</head><body class="present">\n'
    )


def render_html(
    records: list[dict],
    bash_output_lines: int,
    title: str,
    mode: str = "curate",
) -> str:
    if mode not in {"curate", "present"}:
        raise ValueError(f"mode must be 'curate' or 'present', got {mode!r}")

    toolbar = CURATE_TOOLBAR if mode == "curate" else PRESENT_TOOLBAR
    script = CURATE_SCRIPT if mode == "curate" else PRESENT_SCRIPT

    out: list[str] = [
        _head(title),
        toolbar,
        '<div class="progress" id="progress" style="width:0%"></div>\n',
        f"<h1>{_esc(title)}</h1>\n",
    ]

    pending: dict[str, dict] = {}

    for rec in records:
        rtype = rec.get("type")
        msg = rec.get("message") or {}
        content = msg.get("content")

        if rtype == "user":
            for tr in tool_results(content):
                tid = tr.get("tool_use_id", "")
                tcontent = tr.get("content", "")
                if tid in pending and isinstance(tcontent, str) and "has answered" in tcontent:
                    q_input = pending.pop(tid)
                    answers = dict(parse_answer_string(tcontent))
                    out.append('<div class="msg answers step">')
                    out.append('<div class="role">User selected</div>')
                    for q in q_input.get("questions", []):
                        qtext = q.get("question", "")
                        out.append('<div class="question-card">')
                        out.append(f'<div class="q">{_esc(qtext)}</div>')
                        out.append('<div class="options">')
                        selected = answers.get(qtext, "")
                        selected_set = {s.strip() for s in selected.split(",")}
                        for opt in q.get("options", []):
                            label = opt.get("label", "")
                            desc = opt.get("description", "")
                            cls = "option selected" if label in selected_set else "option"
                            out.append(
                                f'<div class="{cls}"><div class="label">{_esc(label)}</div>'
                                f'<div class="desc">{_esc(desc)}</div></div>'
                            )
                        out.append("</div></div>")
                    out.append("</div>")

            for txt in text_blocks(content):
                txt = clean_user_text(txt)
                if not txt:
                    continue
                slash = SLASH_CMD_RE.search(txt)
                if slash:
                    out.append(
                        f'<h2 class="user step">User invoked /{_esc(slash.group(1).strip().lstrip("/"))}</h2>'
                    )
                    continue
                if looks_like_slash_command_expansion(txt):
                    out.append('<div class="skipped">[slash-command body omitted]</div>')
                    continue
                out.append('<div class="msg user step">')
                out.append('<div class="role">User</div>')
                out.append(f"<div>{_esc(txt).replace(chr(10), '<br>')}</div>")
                out.append("</div>")

        elif rtype == "assistant":
            texts = [t for t in text_blocks(content) if t.strip()]
            tools = tool_uses(content)
            for t in texts:
                out.append('<div class="msg assistant step">')
                out.append('<div class="role">Assistant</div>')
                out.append(f"<div>{_esc(t.rstrip()).replace(chr(10), '<br>')}</div>")
                out.append("</div>")
            for tu in tools:
                name = tu.get("name", "")
                inp = tu.get("input", {})
                if name == "AskUserQuestion":
                    pending[tu.get("id", "")] = inp
                    out.append('<div class="msg tool step">')
                    out.append('<div class="tool-name">Assistant asked</div>')
                    for q in inp.get("questions", []):
                        out.append('<div class="question-card">')
                        out.append(f'<div class="q">{_esc(q.get("question", ""))}</div>')
                        out.append('<div class="options">')
                        for opt in q.get("options", []):
                            label = opt.get("label", "")
                            desc = opt.get("description", "")
                            out.append(
                                f'<div class="option"><div class="label">{_esc(label)}</div>'
                                f'<div class="desc">{_esc(desc)}</div></div>'
                            )
                        out.append("</div></div>")
                    out.append("</div>")
                elif name == "Bash":
                    cmd = inp.get("command", "")
                    desc = inp.get("description", "")
                    out.append('<div class="msg tool step">')
                    out.append(f'<div class="tool-name">Bash \u2014 {_esc(desc)}</div>')
                    out.append(f"<pre><code>{_esc(cmd.strip())}</code></pre>")
                    raw = _find_tool_output(records, tu.get("id", ""))
                    if raw:
                        snipped, truncated = _snippet(raw, bash_output_lines)
                        if snipped.strip():
                            out.append(f"<pre><code>{_esc(snipped)}</code></pre>")
                            if truncated:
                                out.append('<div class="truncated">(output truncated)</div>')
                    out.append("</div>")
                else:
                    summary = _summarize_input(name, inp)
                    out.append(
                        f'<div class="msg tool step"><div class="tool-name">{_esc(name)}</div>'
                        f"<div>{_esc(summary)}</div></div>"
                    )

    out.append(script)
    out.append("</body></html>\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("session", type=Path, help="Path to <session>.jsonl")
    ap.add_argument(
        "--format",
        choices=["md", "html", "both"],
        default="both",
        help="Output format (default: both)",
    )
    ap.add_argument(
        "--mode",
        choices=["curate", "present"],
        default="curate",
        help="HTML mode: curate (keep/skip + export) or present (PNG export). Default: curate",
    )
    ap.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    ap.add_argument(
        "--bash-output-lines",
        type=int,
        default=20,
        help="Max lines of Bash output per tool call (default: 20, 0 = none)",
    )
    ap.add_argument("--title", default=None, help="Title (default: session filename)")
    args = ap.parse_args(argv)

    if not args.session.exists():
        ap.error(f"session file not found: {args.session}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_records(args.session))
    title = args.title or f"Transcript \u2014 {args.session.stem}"
    stem = args.session.stem

    if args.format in {"md", "both"}:
        md = render_markdown(records, args.bash_output_lines)
        out_md = args.output_dir / f"{stem}.md"
        out_md.write_text(md)
        print(f"Wrote {out_md} ({len(md):,} bytes)")

    if args.format in {"html", "both"}:
        h = render_html(records, args.bash_output_lines, title, mode=args.mode)
        suffix = "" if args.mode == "curate" else "_present"
        out_html = args.output_dir / f"{stem}{suffix}.html"
        out_html.write_text(h)
        print(f"Wrote {out_html} ({len(h):,} bytes, mode={args.mode})")


if __name__ == "__main__":
    main()
