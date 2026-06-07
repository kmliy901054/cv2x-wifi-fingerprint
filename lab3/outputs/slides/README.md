# Lab 3 presentation deck

A journey-style talk: how the WiFi indoor-localization model went from a
1.57 m KNN baseline to a 0.79 m coarse-to-fine cascade, and how it held up
live in the lab. Slides are in English; speaker notes are in Traditional
Chinese (in the PowerPoint notes pane).

## Files

| File | Purpose |
|---|---|
| **`lab3_presentation.pptx`** | **The course deck (6/08) — 33 slides, present this one** |
| `PRESENTATION_SCRIPT.md` | Read-along: every slide's title/bullets + long zh-TW speaker script |
| `deck_v3.json` | Content spec for the 33-slide deck (current) |
| `build_pptx.py` | Generator: JSON → `.pptx` (figures, notes, footer, ▶ video placeholder) |
| `make_script_md.py` | Generator: `python make_script_md.py deck_v3.json` → script |
| `deck_v2.json` | Previous 26-slide spec |
| `lab3_journey.pptx` / `deck_final.json` | Earliest 18-slide journey deck |

The 33-slide deck = the 26-slide deck + 7 depth slides (anatomy of one scan, how
data was collected, how the Set Transformer reads a set, how GP-kriging fabricates
coverage, the training recipe, why predict a distribution, nested-CV methodology),
with substantially longer zh-TW speaker notes on every slide. Design borrows from
open-slide: a footer on every content slide (`deck title · section · n / N`),
chapter-style section dividers, and a hero title.

Required-section dividers (題目定義 / 做法 / 資料集切分 / 實驗結果), an
**honesty-audit** slide (0.650 m was test-set overfitting; **0.752 m** honest
headline; ~0.94 m strict nested-CV), and a reserved **live-demo video slide**
(▶ placeholder — Insert → Video over it) are all present.

Built by multi-agent workflows: section drafters → professor + audience + design
reviewers → an editor agent that folded the feedback in.

## Rebuild

```bash
# from lab3/outputs/slides   (close the .pptx in PowerPoint first — it locks the file)
python build_pptx.py deck_v3.json lab3_presentation.pptx
python make_script_md.py deck_v3.json
soffice --headless --convert-to pdf lab3_presentation.pptx   # optional flat PDF
```

Edit `deck_v3.json` (text/notes/figure) and re-run. Figures come from `../figures/`.
To embed the demo video: open the deck, go to the live-demo slide, Insert → Video
over the ▶ placeholder image.

## How the deck was made

1. Draft written by hand into `deck_spec.json` (18 slides, assertion titles).
2. `optimize_deck.workflow.js` ran 5 rounds: each round a panel of five critics
   (narrative, technical accuracy, tone, slide design, speaker notes) reviewed the
   deck, then an editor agent folded the feedback into a revised version. ~76 edits
   total. Output saved as `deck_final.json`.
3. `build_pptx.py` rendered the final spec to `.pptx`, embedding figures and
   placing the Chinese notes in the speaker-notes pane.
