# Lab 3 presentation deck

A journey-style talk: how the WiFi indoor-localization model went from a
1.57 m KNN baseline to a 0.79 m coarse-to-fine cascade, and how it held up
live in the lab. Slides are in English; speaker notes are in Traditional
Chinese (in the PowerPoint notes pane).

## Files

| File | Purpose |
|---|---|
| `lab3_journey.pptx` | The deck — 18 slides, editable in PowerPoint / Keynote / Google Slides |
| `lab3_journey.pdf` | Flat render for quick viewing / projecting |
| `deck_final.json` | Content spec (titles, bullets, figure per slide, zh-TW notes) |
| `deck_spec.json` | The hand-written first draft, before optimisation |
| `build_pptx.py` | Generator: JSON spec → `.pptx` |
| `optimize_deck.workflow.js` | The 5-round critic/reviser workflow used to polish the draft |

## Rebuild

```bash
# from lab3/outputs/slides
python3 build_pptx.py deck_final.json lab3_journey.pptx
# optional flat render (needs libreoffice)
soffice --headless --convert-to pdf lab3_journey.pptx
```

Edit `deck_final.json` (text/notes/figure) and re-run `build_pptx.py` to
regenerate. Figures are pulled from `../figures/`.

## How the deck was made

1. Draft written by hand into `deck_spec.json` (18 slides, assertion titles).
2. `optimize_deck.workflow.js` ran 5 rounds: each round a panel of five critics
   (narrative, technical accuracy, tone, slide design, speaker notes) reviewed the
   deck, then an editor agent folded the feedback into a revised version. ~76 edits
   total. Output saved as `deck_final.json`.
3. `build_pptx.py` rendered the final spec to `.pptx`, embedding figures and
   placing the Chinese notes in the speaker-notes pane.
