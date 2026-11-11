# NimbusPay Ticket Triage — LoRA fine-tune

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` — 1.54B params, under the 4B cap
**Loaded in 4-bit?** **No** — the final adapter is 16-bit LoRA, and that is a measured choice, not
a default. Experiment **E2a** ran the same configuration in 4-bit QLoRA: it scored **82.5%** against
16-bit's 90.0% on the same data, saved only **0.74 GB** of peak memory, and took **4 minutes longer**.
At 1.5B the base already fits a T4 in 11.82 GB, so quantization buys headroom this model does not need
and costs 7.5 points of exact match to get it.
**Trainable parameters:** 18.46M (`r=16`, all attention + MLP projections) — 36.9% of the 50M cap.

**Best result:** dev **exact match 94.0%**, **mean field accuracy 98.9%**, **JSON validity 100%** (n=200),
up from 0.0% / 0.0% / 0.0% for the un-tuned base model on the same fixed prompt. Trained in **51.5 min**
on a free T4 at **12.00 GB** peak. All 400 test predictions parse with the correct key set, and the saved
adapter reloaded from disk reproduces 12/12 sampled rows exactly.

The submitted model is the **E6** configuration — the cleaned 1,350 rows plus **720 counterfactual
minimal pairs** I generated, which was the best arm in Part D. Against the E0 reference (cleaned data
alone, 90.0%) that is **+4.0 points**, and `priority` rises 91.5% → 96.0%.

---

## Status

Parts A (audit), B (baselines), C (fine-tuning), D (experiments) and E (error analysis) are complete,
and the hidden-test predictions are generated and verified.

**Part D has six arms, covering all four mandated comparisons.** `report.md` §4 and
`experiments_results.json` carry the table: E0 (reference), E1a (raw vs cleaned data), E2a (16-bit vs
4-bit QLoRA), E3a (epochs), and E5 + E6 (my two own ideas). E2a was measured on a **Kaggle** Tesla T4
after the Colab daily GPU quota ran out — same GPU, one device pinned with `CUDA_VISIBLE_DEVICES=0`
so its memory and time figures stay comparable to the other arms. The notebook resolves its paths
from a `WORK` root and runs unchanged on either host.

The final-model pass uses `RUN_EXPERIMENTS = False`, which completes everything except Part D in
~47 min inside one free-tier session. The reference arm E0 *is* the final run reported above.

## Layout

| Path | What it is |
|---|---|
| `notebook.ipynb` | The whole pipeline, **with outputs** from the single run that produced every number here: audit → baselines → fine-tune → error analysis → test predictions. It is the executed copy from that Kaggle run; the one Kaggle-only cell that extracted it (Kaggle hides files named `__notebook__.ipynb`) has been removed, so all 29 code cells are pipeline code and every one carries its real output. Paths resolve from a `WORK` root, so it runs unchanged on Colab or Kaggle |
| `report.pdf` | **The 3-page report** (page count verified by rendering, not estimated) |
| `report.md` | Same content as Markdown, for reading in the repo; `report.css` is the stylesheet used to render the PDF |
| `cleaning_log.csv` | 388 rows of `id,action,reason` — every row dropped or fixed |
| `src/audit.py` | The audit + cleaner as a standalone script, runnable with no GPU and no dependencies |
| `adapter/` | Final PEFT adapter — `adapter_config.json` + `adapter_model.safetensors` (70.5 MB, under GitHub's 100 MB limit, so no Git LFS) |
| `predictions_test.jsonl` | 400 lines of `{"id","output"}`, all parsing with the correct 8-key set |
| `experiments_results.json` | The Part D table as data — one record per arm, including E2a marked not-completed |
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
   dev scoring, the error analysis and the test predictions. Measured end-to-end: **~1 h 8 min** on a
   free T4 (install, audit <1 min, baselines ~2.5 min each, training 51.5 min on the 2,070-row
   augmented set, dev eval ~3 min, 400 test predictions ~7 min) — inside the 3-hour budget.
4. The last cell zips everything to `$WORK/nimbuspay_artifacts.zip` (~91 MB). Download it from the
   Files panel — **artifacts are VM-local and are lost when the runtime recycles.**
5. The Part D table comes from setting `RUN_EXPERIMENTS = True` and running again. It checkpoints
   `experiments_results.json` after every arm, so a disconnect loses at most the run in flight —
   which is exactly what happened to E2a. Then set it back to `False`.

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
**Measured: 89.0% vs E0's 90.0% — no effect.** The amount is in the *input*, which attention reads
from every output position, so reordering buys nothing when the value only has to be looked up
rather than derived. Reported as a negative result; it is what sharpened the diagnosis behind E6.

**Counterfactual minimal pairs.** The whole decision surface rests on thinly-taught triggers (the
₹5,000 boundary: 61 rows; ₹50,000: 45; legal threats: 87; quoted `>` history: 38 rows against 11 of
400 test rows). Since the rule engine is exact on dev it serves as a **label oracle** — mutate one
trigger in a real ticket, recompute the label deterministically. 720 generated rows, all passing the
validator dev passes, zero independent re-derivation mismatches; threshold coverage 106 → 264 rows.
Experiment E6, and **this is the submitted model**: retrained as the final run it scores **94.0% exact
/ 98.9% field**, against the E0 reference's 90.0% / 97.9%. The targeted error class moved with it —
rupee-threshold mistakes fall from 9 of 20 root causes to 4 of 12 (`report.md` §3).
