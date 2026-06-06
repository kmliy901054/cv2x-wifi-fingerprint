# Lab 3 presentation deck

A journey-style talk: how the WiFi indoor-localization model went from a
1.57 m KNN baseline to a 0.79 m coarse-to-fine cascade, and how it held up
live in the lab. Slides are in English; speaker notes are in Traditional
Chinese (in the PowerPoint notes pane).

## Files

| File | Purpose |
|---|---|
| **`lab3_presentation.pptx`** | **The course deck (6/08) — 26 slides, present this one** |
| `PRESENTATION_SCRIPT.md` | Read-along: every slide's title/bullets + zh-TW speaker script |
| `deck_v2.json` | Content spec for the 26-slide deck |
| `build_pptx.py` | Generator: JSON spec → `.pptx` (figures, notes, ▶ video placeholder) |
| `make_script_md.py` | Generator: deck JSON → `PRESENTATION_SCRIPT.md` |
| `lab3_journey.pptx` / `deck_final.json` | Earlier 18-slide journey deck (superseded) |

The 26-slide deck adds: explicit section dividers for the four required parts
(題目定義 / 做法 / 資料集切分 / 實驗結果), a per-split results slide, an
**honesty-audit** slide (0.650 m was test-set overfitting; **0.752 m** is the
honest headline; ~0.95 m strict nested-CV), and a reserved **live-demo video
slide** (▶ placeholder — drop the recording onto that slide: Insert → Video).

Built by a multi-agent workflow: 5 section drafters → professor + audience +
design reviewers → an editor agent that folded the feedback in.

## Rebuild

```bash
# from lab3/outputs/slides
python3 build_pptx.py deck_v2.json lab3_presentation.pptx
python3 make_script_md.py
# optional flat PDF render (needs libreoffice)
soffice --headless --convert-to pdf lab3_presentation.pptx
```

Edit `deck_v2.json` (text/notes/figure) and re-run `build_pptx.py`. Figures are
pulled from `../figures/`. To embed the demo video: open the deck, go to the
live-demo slide, Insert → Video over the ▶ placeholder image.

## How the deck was made

1. Draft written by hand into `deck_spec.json` (18 slides, assertion titles).
2. `optimize_deck.workflow.js` ran 5 rounds: each round a panel of five critics
   (narrative, technical accuracy, tone, slide design, speaker notes) reviewed the
   deck, then an editor agent folded the feedback into a revised version. ~76 edits
   total. Output saved as `deck_final.json`.
3. `build_pptx.py` rendered the final spec to `.pptx`, embedding figures and
   placing the Chinese notes in the speaker-notes pane.
