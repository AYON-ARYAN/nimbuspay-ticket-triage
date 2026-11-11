# NimbusPay Ticket Triage — LoRA fine-tune

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` — 1.54B params, under the 4B cap
**Loaded in 4-bit?** **No** — the final adapter is 16-bit LoRA. A 4-bit QLoRA arm is *coded* as
experiment E2 but was **not run** in this pass (see "Status" below); the base model is small enough
that 16-bit already fits a T4 at 11.82 GB peak, so QLoRA would buy headroom rather than feasibility.
**Trainable parameters:** 18.46M (`r=16`, all attention + MLP projections) — 36.9% of the 50M cap.

**Best result:** dev **exact match 90.0%**, **mean field accuracy 97.9%**, **JSON validity 100%** (n=200),
up from 0.0% / 0.0% / 0.0% for the un-tuned base model on the same fixed prompt. Trained in **19.9 min**
on a free Colab T4 at **11.82 GB** peak. All 400 test predictions parse with the correct key set, and the
saved adapter reloaded from disk reproduces 12/12 sampled rows exactly.

---

## Status

Parts A (audit), B (baselines), C (fine-tuning) and E (error analysis) are complete, and the hidden-test
predictions are generated and verified. **Part D (controlled experiments) is not filled in.** This run
used `RUN_EXPERIMENTS = False`, which completes everything else in ~47 min inside one free-tier session.
The reference arm E0 *is* the final run reported above; the other seven arms are implemented and run
from a single flag, needing roughly 2.5 h more T4 time. `report.md` §4 states this plainly rather than
implying the table exists.

## Layout

| Path | What it is |
|---|---|
| `notebook.ipynb` | The whole pipeline, **with outputs** from the run that produced every number here: audit → baselines → fine-tune → error analysis → test predictions |
| `report.pdf` | **The 3-page report** (page count verified by rendering, not estimated) |
| `report.md` | Same content as Markdown, for reading in the repo; `report.css` is the stylesheet used to render the PDF |
| `cleaning_log.csv` | 388 rows of `id,action,reason` — every row dropped or fixed |
| `src/audit.py` | The audit + cleaner as a standalone script, runnable with no GPU and no dependencies |
| `adapter/` | Final PEFT adapter — `adapter_config.json` + `adapter_model.safetensors` (70.5 MB, under GitHub's 100 MB limit, so no Git LFS) |
| `predictions_test.jsonl` | 400 lines of `{"id","output"}`, all parsing with the correct 8-key set |
| `final_metrics.json`, `baselines.json`, `error_attribution.json`, `dev_predictions.jsonl` | Raw numbers behind every figure in the report |

## Reproducing

The audit half needs no GPU and no third-party packages:

```bash
python3 src/audit.py --data ../data --out build
# -> build/train_clean.jsonl, build/cleaning_log.csv
```

It asserts that the validator emits **zero flags on all 200 dev rows** before it will touch
`train.jsonl`, and asserts the cleaned output is spec-clean afterwards. Both assertions pass.

The training half needs a T4:

1. Open `notebook.ipynb` in Colab, set **Runtime → Change runtime type → T4 GPU**, and connect.
2. Upload `train.jsonl`, `dev.jsonl` and `test_inputs.jsonl` into `/content` using the **Files**
   panel on the left. Nothing touches Google Drive: the notebook reads from `/content` and writes
   every artifact to `/content/out`. (If the files are missing, the data cell falls back to an
   interactive upload prompt and then asserts all three are present before continuing.)
3. **Run all.** With `RUN_EXPERIMENTS = False` this runs the audit, both baselines, the final model,
   dev scoring, the error analysis and the test predictions. Measured end-to-end: **~47 min** on a
   free T4 (install 16 s, audit <1 min, baselines 2.5 min each, training 19.9 min, dev eval 3 min,
   400 test predictions 6.6 min) — well inside the 3-hour budget.
4. The last cell zips everything to `/content/nimbuspay_artifacts.zip` (~72 MB). Download it from the
   Files panel — **artifacts are VM-local and are lost when the runtime recycles.**
5. To produce the Part D table, set `RUN_EXPERIMENTS = True` and run once more. It trains seven
   configurations and checkpoints `experiments_results.json` after each, so a disconnect loses at
   most one run. Then set it back to `False`.

## Inference contract

The notebook generates exactly the way the brief specifies, and nothing in the pipeline depends on
anything else:

* the `system` and `user` messages **verbatim** from the data, via Qwen's own chat template;
* `model.generate`, greedy (`do_sample=False, num_beams=1`), `max_new_tokens=200`;
* `.strip()` as the **only** post-processing — no JSON repair, no field filling, no constrained
  decoding, no second model.

Left padding is used for batched generation so a batch is bit-identical to one-at-a-time decoding.
The last cell reloads the saved adapter from disk and re-generates 12 random test rows to confirm
they match `predictions_test.jsonl` — the same check the graders described.

## Headline audit finding

70 training rows carry the key `prio` with values `low`/`medium`/`high` rather than `priority` with
`P1`/`P2`/`P3`. They are not noise — they are a **v1 spec export** (`SCHEMA.md` calls itself "v2,
current"), and the mapping to what the v2 ladder derives is perfect with zero exceptions. Recomputing
those labels, which is the obvious move, would have thrown away 70 correct ones; the right fix is a
value remap. The 30 `DD/MM/YYYY` dates fall entirely inside that same batch, which is what confirmed
a single legacy exporter rather than three unrelated faults.

Full detail, including two hypotheses I tested and rejected, is in `report.md` §1.

## Two things that are not standard practice

**Dependency-ordered output keys.** `score.py` scores by `pred[k]` lookup and `set(pred) == set(KEYS)`,
so key *order* is not scored — verified against the real scorer. The schema's order puts `priority`
*before* the `amount_paise` it depends on, forcing the model to apply two rupee thresholds without
writing the amount down. Training on
`category → amount_paise → priority → needs_human → ...` lets each rule-derived field attend to its
own inputs: chain-of-thought inside a fixed 8-key schema, at zero token cost. Experiment E5.

**Counterfactual minimal pairs.** The whole decision surface rests on thinly-taught triggers (the
₹5,000 boundary: 61 rows; ₹50,000: 45; legal threats: 87; quoted `>` history: 38 rows against 11 of
400 test rows). Since the rule engine is exact on dev it serves as a **label oracle** — mutate one
trigger in a real ticket, recompute the label deterministically. 720 generated rows, all passing the
validator dev passes, zero independent re-derivation mismatches; threshold coverage 106 → 264 rows.
Experiment E6.
