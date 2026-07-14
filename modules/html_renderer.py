# html_renderer.py
from typing import List, Optional
from jinja2 import Template
from modules.pdf_extractor import Page, Block, Line, Span
import html as html_mod

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
        .page { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .block { margin-bottom: 10px; }
        .block-heading { font-weight: bold; }
        .block-paragraph { line-height: 1.6; }
        .block-metadata { color: #666; font-size: 0.9em; }
        .block-reference { color: #888; font-size: 0.85em; margin-left: 16px; }
        .block-table { border-collapse: collapse; width: 100%; }
        .block-table td, .block-table th { border: 1px solid #ccc; padding: 6px; }
        .block-figure { text-align: center; }
        .block-figure img { max-width: 100%; }
        .block-figure figcaption { font-style: italic; }
        .bold { font-weight: bold; }
        .italic { font-style: italic; }
        .translation { color: #333; }
        .original { color: #999; font-size: 0.9em; }
        details.original-block { margin: 4px 0; }
        details.original-block summary { color: #888; font-size: 0.85em; cursor: pointer; display: inline; }
        details.original-block summary:hover { color: #555; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    {% for page in pages %}
    <div class="page">
        <h3>Страница {{ page.num }}</h3>
        {% for block in page.blocks %}
            {% if block.type == 'figure' %}
                <figure class="block-figure">
                    <img src="data:image/png;base64,{{ block.image_data }}" />
                    {% if block.caption %}
                    <figcaption>{{ block.caption }}</figcaption>
                    {% endif %}
                </figure>
            {% elif block.type == 'table' and block.table_data %}
                <table class="block-table">
                {% for row in block.table_data %}
                    <tr>
                    {% for cell in row %}
                        <td>{{ cell }}</td>
                    {% endfor %}
                    </tr>
                {% endfor %}
                </table>
            {% elif block.type == 'heading' %}
                <div class="block-heading">{{ render_block(block) }}</div>
            {% elif block.type == 'reference_heading' %}
                <h2 class="block-heading">{{ render_block(block) }}</h2>
            {% elif block.type == 'metadata' %}
                <div class="block-metadata">{{ render_block(block) }}</div>
            {% elif block.type == 'reference' %}
                <div class="block-reference">{{ render_block(block) }}</div>
            {% else %}
                <p class="block-paragraph">{{ render_block(block) }}</p>
            {% endif %}
        {% endfor %}
    </div>
    {% endfor %}
</body>
</html>
"""

SHORT_THRESHOLD = 40

def render_block(block):
    orig = render_spans(block)
    trans = block.translation
    if not trans:
        return orig
    orig_len = len(orig.strip())
    if orig_len <= SHORT_THRESHOLD:
        return f'<span class="translation">{html_mod.escape(trans)}</span>'
    escaped_orig = html_mod.escape(orig)
    escaped_trans = html_mod.escape(trans)
    return f'{escaped_trans}<details class="original-block"><summary>Оригинал</summary><span class="original">{escaped_orig}</span></details>'

def render_spans(block):
    parts = []
    for line in block.lines:
        for span in line.spans:
            text = span.text
            if span.flags & 2**0:
                text = f"<b>{text}</b>"
            if span.flags & 2**1:
                text = f"<i>{text}</i>"
            parts.append(text)
        parts.append("\n")
    return " ".join(parts)

def generate_html(pages: List[Page], title: str, output_path: str):
    template = Template(HTML_TEMPLATE)
    rendered = template.render(pages=pages, title=title, render_block=render_block)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered)
