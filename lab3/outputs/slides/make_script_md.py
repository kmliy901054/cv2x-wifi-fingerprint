"""Render deck_v2.json into a human-readable bilingual speaker script."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / 'deck_v2.json', encoding='utf-8'))
m = d['meta']
L = []
L.append(f"# {m['title']}\n")
L.append(f"> {m['subtitle']}  ")
L.append(f"> {m['author']}\n")
L.append(f"英文投影片標題/條列 + 中文口稿。共 {len(d['slides'])} 張投影片。\n")
L.append("可開 `lab3_presentation.pptx` 編輯;這份是給你對稿/排練用的文字版。\n")
L.append("---\n")
for i, s in enumerate(d['slides'], 1):
    L.append(f"## {i}. {s['title']}")
    if s.get('subtitle'):
        L.append(f"*{s['subtitle']}*")
    L.append("")
    for b in s.get('bullets', []):
        L.append(f"- {b}")
    if s.get('bullets'):
        L.append("")
    if s.get('figure'):
        vid = "  ▶ **影片占位 — 換成 live demo 錄影**" if s.get('video_placeholder') else ""
        L.append(f"圖: `outputs/figures/{s['figure']}`{vid}")
        L.append("")
    L.append(f"口稿: {s['notes']}")
    L.append("\n---\n")
(HERE / 'PRESENTATION_SCRIPT.md').write_text('\n'.join(L), encoding='utf-8')
print(f"wrote PRESENTATION_SCRIPT.md ({len(d['slides'])} slides)")
