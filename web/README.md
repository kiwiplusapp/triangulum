# Vault Console

A Next.js console for the Vault agent: signals, calibration, the learning
stack, and the capital gate — on a board where every panel can be dragged,
resized, hidden and restored, with the arrangement persisted per browser.

```bash
# 1. the backend, with a seeded track record and a completed learning run
python3 -m vault --fixtures demo --agent oracle --calls 160 --learn

# 2. the console
cd web && npm install && npm run dev     # http://localhost:3000
```

The console proxies `/api/*` to `http://127.0.0.1:8899` by default. Point it
somewhere else with `VAULT_API=http://host:port npm run dev`.

## Design

**Near-monochrome, deliberately.** Eleven greys, one cold accent, and two
heavily desaturated tints for sign. Everything a more colourful dashboard
would encode in hue is encoded here in luminance, weight and space.

The reason is not taste. On a dashboard where twelve things are brightly
coloured, nothing is left to say *look at this one* — every element is already
shouting, so the genuinely urgent state cannot be distinguished from the
merely present. Holding almost everything to greyscale means the accent, used
on perhaps two elements per screen, actually directs the eye.

**The board does not rearrange itself.** Panels may overlap; nothing
auto-packs. A grid that reflows while you are dragging makes precise
arrangement impossible — you move one panel and three others jump. The
arrangement is the user's, so the user's arrangement wins.

Drag by the header, resize from the corner, or focus a header and use the
arrow keys (shift+arrows resizes). Layout is stored in `localStorage` under
`vault.board.v1` and merged forward on load, so a stored layout from an
older version keeps its positions and still gains any panel added since.

## The rule every panel follows

**Never render a number the backend did not supply.** Every panel has an
explicit empty state naming what is missing and how to produce it. A dashboard
that shows `0.0%` for "no data" and `0.0%` for "measured zero" is worse than
one that shows nothing, because it is confidently wrong rather than obviously
incomplete.

The same applies to the backend being down: that renders as an offline banner
naming the command to start it, not as a console full of zeros.

## Panels

| Panel | What it shows |
|---|---|
| Track record | Resolved calls, hit rate with both confidence bounds, Brier against the 0.25 baseline, capital released |
| Capital gate | The four gates, each with its requirement and current value |
| Quantitative prior | P(up), its distance from the base rate, and the weight split across signals / model / base rate |
| Macro regime | Growth-inflation quadrant probabilities, scores, and input staleness |
| Signal readings | All 23 signals, sortable, with unusable ones separated rather than zeroed |
| Signal scorecard | Per-signal IC, hit rate, both-sides counts, weight, and the verdict for every rejection |
| Walk-forward training | Per-fold Brier for base rate, logistic and network, and which won |
| Provenance | Permutation importance, including features whose removal would improve the model |
| Open calls / Call history | The journal |

Two columns are worth explaining because they are the ones that catch
mistakes:

- **`+/−`** in the scorecard is how many times a signal read positive versus
  negative. A signal showing `177/0` never took the other side, so its hit
  rate is a restatement of the market's own drift — it is rejected as
  `one-sided`, and the column is there so you can see why at a glance.
- **"Nothing below an IC of X is distinguishable from zero at this sample
  size"** is the binding constraint, and it is usually invisible. With two
  years of history at a 21-day horizon there are about 34 independent
  observations and X is around 0.5, while real macro signals live at
  0.02–0.06. The honest reading is not "no signal works" but "this sample
  cannot see a signal of realistic size".

## Dependencies

React, React DOM and Next. Thirty packages total. Drag and resize are about
150 lines of pointer arithmetic in `lib/layout.ts` and `components/Board.tsx`
rather than a grid library, which would have been a larger dependency than the
rest of the app put together.
