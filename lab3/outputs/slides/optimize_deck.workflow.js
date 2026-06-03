export const meta = {
  name: 'optimize-lab3-deck',
  description: 'Five rounds of critic-panel + reviser to polish the Lab 3 journey slide deck',
  phases: [
    { title: 'Round 1' }, { title: 'Round 2' }, { title: 'Round 3' },
    { title: 'Round 4' }, { title: 'Round 5' },
  ],
}

// The deck (English slides + Traditional-Chinese speaker notes) comes from `args`,
// or is loaded from disk by a loader agent if args is not supplied.
let deck = args
const DECK_PATH = '/home/wayne/lab2_submit_FINAL_20260523/lab3/outputs/slides/deck_spec.json'

// ---- Ground truth the agents must respect (no invented numbers) -------------
const FACTS = `
AUTHORITATIVE NUMBERS (Split A random 80/20 test, 363 samples, median error in metres):
  KNN k=5 = 1.568 | MLP = 1.302 | MaskedMDN = 1.371 | Set Transformer MDN = 1.093
  Set Transformer Big x5-ens = 1.083 | + GP synth (single) = 0.906 | + GP synth x5-ens = 0.889
  C-Mixup x5-ens = 0.942 | Heatmap x5-ens = 0.883 | Cascade x5-ens = 0.793 (WINNER)
  CNN xattn x5-ens = 0.907 | 3-Cascade x10-ens = 0.800 | Diffusion x5-ens = 1.821
  Cascade winner: median 0.793, mean 1.117, p90 2.559, 27% within 0.3 m.
SYNTHETIC ABLATION: Split A real-only 1.119 -> real+synth 0.906 (-19%);
  Split C (morning->evening) real-only 1.726 -> real+synth 1.811 (+5%, slightly worse).
DATA: 1812 records (morning 912 + evening 900); 115 unique BSSID, 80 kept (>=10 hits); 189.5 m^2.
  Within-cell RSSI std ~3.5 dBm; morning vs evening median |drift| ~0.8 dBm.
LIVE DEMO (in lab, ESP32 over USB + ROS2/RViz): 80-95% APs matched; standing-still spread <0.1 m;
  single ground-truth check 0.4-0.8 m, consistent with offline median. ESP32 scan period ~3.8 s.
CONSTRAINTS: slide text MUST be English; speaker notes MUST be Traditional Chinese (zh-TW).
Do NOT introduce any number not listed above. Keep every figure filename exactly as given.`

const FIG_DESC = `
FIGURE INVENTORY (what each filename shows — do not invent new files):
  demo_snapshot.png        one live prediction: heatmap + true(green X)/pred(blue dot) on floor plan
  data_coverage.png        train(blue)+test(red) positions on floor plan; upper region sparse
  error_vs_aps_drift.png   left: error vs #matched APs; right: morning vs evening RSSI scatter
  architectures/arch_knn.png            KNN pipeline block diagram
  architectures/arch_masked_mdn.png     MaskedMDN block diagram
  architectures/arch_set_transformer_mdn.png  Set Transformer + MDN block diagram
  synth_ablation.png       left: CDF real vs real+synth (Split A); right: median bars Split A & C
  architectures/arch_heatmap.png        heatmap classification head block diagram
  architectures/arch_cascade.png        coarse->fine cascade block diagram (the winner)
  gating_before_after.png  3 panels: fine alone (multi-modal) | coarse gate | gated (single peak)
  error_boxplot.png        per-model error distribution incl. failed models
  ladder_bar.png           median error bar chart across all experiments (green=winner,red=failed)
  error_cdf.png            CDF of all major models
  region_reliability.png   per-cell mean error vs sample density (two panels)
  cascade_random100.png    100 random predictions, error vectors on floor plan`

const LENSES = [
  { key: 'narrative', brief:
    'STORY COHERENCE. Read slide titles in order — do they tell the whole journey by themselves (vertical alignment)? Is the motivation->obstacle->decision->result arc clear? Are transitions between turning points logically motivated (does each step explain WHY the previous one was not enough)? Flag any slide that breaks the arc or repeats.' },
  { key: 'accuracy', brief:
    'TECHNICAL ACCURACY. Check every number and claim against the AUTHORITATIVE NUMBERS. Flag any figure-vs-text mismatch, any overclaim, any unsupported causal claim, any number not in the facts. Verify the figure on each slide actually supports that slide\'s assertion.' },
  { key: 'tone', brief:
    'TONE / NO-AI-FLAVOR. English slides must read like a competent grad student wrote them: plain, concrete, objective. Flag hype words (powerful, seamless, leverage, revolutionary, cutting-edge, robustly), empty intensifiers, marketing voice, over-hedging, and em-dash overuse. Chinese notes must sound like natural spoken Taiwanese Mandarin a presenter would actually say — flag stiff/translated-sounding phrasing and any AI cliche.' },
  { key: 'design', brief:
    'SLIDE DESIGN / COGNITIVE LOAD. One idea per slide. Titles should be assertions (a claim), not topics. Bullets: aim 3-4, each short, parallel grammar, no full sentences, no redundancy with the title. Flag slides that are overloaded or whose bullets just restate the title.' },
  { key: 'notes', brief:
    'SPEAKER-NOTE QUALITY. Each note (zh-TW) should be speakable in ~25-40 seconds, cover what the slide does NOT show in text, and not just read the bullets aloud. Flag notes that are too long, too short, mismatched to their slide, or that mention numbers absent from the slide without context.' },
]

const FEEDBACK_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    top_issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          slide_id: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          problem: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['slide_id', 'severity', 'problem', 'fix'],
      },
    },
    overall: { type: 'string' },
  },
  required: ['lens', 'top_issues', 'overall'],
}

const DECK_SCHEMA = {
  type: 'object',
  properties: {
    meta: { type: 'object' },
    slides: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          subtitle: { type: ['string', 'null'] },
          bullets: { type: 'array', items: { type: 'string' } },
          figure: { type: ['string', 'null'] },
          notes: { type: 'string' },
        },
        required: ['id', 'title', 'bullets', 'figure', 'notes'],
      },
    },
    changelog: { type: 'array', items: { type: 'string' } },
  },
  required: ['meta', 'slides'],
}

function criticPrompt(lens, deckJson, round) {
  return `You are reviewing round ${round} of a conference-style slide deck for a university course.
The deck tells the JOURNEY of building a WiFi indoor-localization model (KNN baseline -> ... -> cascade winner).

YOUR LENS: ${lens.brief}

${FACTS}
${FIG_DESC}

Return your single most important findings (max 6, ranked by severity). Be specific: name the slide id,
state the concrete problem, and give a concrete fix (exact replacement wording where you can).
Do NOT rewrite the whole deck — you are only critiquing through your lens. Be terse and concrete; no praise.

CURRENT DECK (JSON):
${deckJson}`
}

function reviserPrompt(deckJson, feedbackJson, round) {
  return `You are the editor producing round ${round} of a slide deck. Apply the critic feedback below to
produce a fully revised deck. Rules:
- Output the COMPLETE deck (all slides), same JSON structure: meta + slides[] (+ changelog[]).
- Slide title/subtitle/bullets MUST be English. notes MUST be Traditional Chinese (zh-TW).
- Make titles assertions (a claim), keep one idea per slide, 3-4 short parallel bullets.
- Keep every "figure" field EXACTLY as in the input (same filenames). Do not invent slides or figures
  unless a critic with HIGH severity explicitly requires reordering; preserve slide ids.
- Use ONLY numbers from the facts. Never invent a statistic.
- Resolve conflicting critic suggestions with editorial judgement; prefer clarity and honesty over hype.
- Strip any AI-flavored or marketing phrasing. Keep speaker notes natural and speakable.
- Put a short bullet list in "changelog" naming the concrete edits you made this round.

${FACTS}
${FIG_DESC}

CURRENT DECK (JSON):
${deckJson}

CRITIC FEEDBACK (JSON array):
${feedbackJson}`
}

// ---- load deck from disk if not passed via args ----------------------------
if (!deck || !Array.isArray(deck.slides)) {
  phase('Round 1')
  deck = await agent(
    `Use the Read tool to read this file: ${DECK_PATH}\n` +
    `Return its JSON content EXACTLY as-is (meta + slides[]). Do not edit, summarize, ` +
    `or translate anything — return the parsed object verbatim.`,
    { label: 'load-deck', phase: 'Round 1', schema: DECK_SCHEMA })
}

// ---- 5 rounds of critique -> revise -----------------------------------------
const history = []
for (let r = 1; r <= 5; r++) {
  phase(`Round ${r}`)
  const deckJson = JSON.stringify(deck, null, 2)

  // panel of critics in parallel
  const feedback = await parallel(LENSES.map(lens => () =>
    agent(criticPrompt(lens, deckJson, r),
      { label: `critic:${lens.key}`, phase: `Round ${r}`, schema: FEEDBACK_SCHEMA })
  ))
  const valid = feedback.filter(Boolean)
  const highCount = valid.reduce((n, f) =>
    n + f.top_issues.filter(i => i.severity === 'high').length, 0)
  log(`Round ${r}: ${valid.length}/5 critics reported, ${highCount} high-severity issues`)

  // reviser folds all feedback into a new deck
  const revised = await agent(
    reviserPrompt(deckJson, JSON.stringify(valid, null, 2), r),
    { label: `reviser:r${r}`, phase: `Round ${r}`, schema: DECK_SCHEMA })

  if (revised && Array.isArray(revised.slides) && revised.slides.length >= 5) {
    deck = { meta: revised.meta || deck.meta, slides: revised.slides }
    history.push({ round: r, high_issues: highCount,
                   changelog: revised.changelog || [] })
    log(`Round ${r} revised: ${(revised.changelog || []).length} edits`)
  } else {
    log(`Round ${r}: reviser returned invalid deck, keeping previous`)
    history.push({ round: r, high_issues: highCount, changelog: ['(revise failed, kept prior)'] })
  }
}

return { deck, history }
