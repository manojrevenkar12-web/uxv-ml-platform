# Results

Measured on 2 CPU cores with the default configuration: 36 simulated flights
(32 s each, 12 day / 12 dusk / 12 night, 14 with fog), ingested into dataset
`uxv@969966aae5426c70`: 5,695 frames in 23 shards (210 MB); 3,793 train,
954 val, 948 test frames; splits are by flight. Raw numbers are in
[`results/benchmarks.json`](results/benchmarks.json); gate reports are in
[`results/`](results/).

Detection metrics are mAP over vehicle / person / building with centre-distance
matching (a prediction matches if its centre is within half the box's short
side, at least 2 px).

Reproduce: `python scripts/train_runs.py WORK`, `python scripts/autolabel_experiment.py WORK`,
`python scripts/benchmarks.py WORK` (WORK containing `sim/` from `uxvml simulate --out WORK/sim`
and `store/` from `uxvml ingest`).

## 1. Ingestion quality checks (scored against simulator truth)

| Check | Injected | Result |
|---|---|---|
| Dead / saturated camera frames | 65 | 65 flagged and dropped, 0 false alarms |
| LiDAR outages > 1 s | 4 flights | 4 flagged, 0 false alarms |
| Wrong-class annotations (physical size rule, using altitude → metres per pixel) | 123 | 83 caught (67%); all 148 flags were real annotation errors |
| LiDAR clock offsets (100–200 ms) | 9 flights | 3 corrected, max error 14 ms; 0 wrong corrections; 6 left "unverified" |

The clock-offset estimator is deliberately conservative. Of the 6 offset
flights it refused to correct (night, fog or low-contrast dusk), 5 of its
estimates were off by more than 30 ms, one with the wrong sign, so applying
them would have made alignment worse. The price is recall: 26 of 36 flights
end up "unverified", including 20 that had no offset. A better estimator for
low-contrast flights is the obvious next step.

## 2. Storage and loading

| Read path (val split, single process, warm page cache) | samples / s |
|---|---|
| Raw logs: decode PNG + read sweep + project LiDAR per frame | 83 |
| Pre-processed, one file per sample | 2,776 |
| Shards (`UXS1`, zero-copy decode) | 3,676 |

Pre-processing at ingestion is the big win (44×). Shards beat per-sample files
by 1.3× here on a warm local disk; their main benefit (one large sequential
request instead of thousands of small ones) shows up on object storage, which
this benchmark does not measure.

Re-ingesting the same flights reproduces the same dataset version id and the
same shard digests (tested); changing one parameter produces a new id.

## 3. Training, reproducibility and fusion

| Model (14 epochs, DDP 2 ranks, Gloo) | test mAP | vehicle | person | building | day | dusk | night |
|---|---|---|---|---|---|---|---|
| camera + LiDAR (`fusion`) | **0.673** | 0.844 | 0.433 | 0.741 | 0.762 | 0.683 | 0.552 |
| camera only | 0.622 | 0.748 | 0.370 | 0.750 | 0.762 | 0.660 | 0.385 |
| fusion, rerun with identical config | 0.673 | identical weights (SHA-256 `bc56bf03…`) |||||
| fusion, trained without sync correction | 0.690 | 0.823 | 0.475 | 0.771 | 0.822 | 0.664 | 0.562 |
| fusion, trusted-vendor labels only (38% fewer frames) | 0.636 | 0.834 | 0.332 | 0.741 | 0.729 | 0.656 | 0.498 |

**Seed-to-seed spread.** The same fusion config trained with seeds 0, 1 and 2:

| seed | test mAP | day | dusk | night |
|---|---|---|---|---|
| 0 | 0.673 | 0.762 | 0.683 | 0.552 |
| 1 | 0.616 | 0.591 | 0.655 | 0.566 |
| 2 | 0.649 | 0.713 | 0.649 | 0.581 |

Overall mAP moves by up to **0.057 between seeds**, which is larger than most
of the single-run differences in the table above. That changes how to read it:

- **Reproducibility.** Two DDP runs with the same config, seed and dataset
  version produced bit-identical weights; different seeds do not, and the
  difference is large.
- **Fusion's robust gain is at night.** Night mAP is 0.55–0.58 for all three
  fusion seeds vs. 0.385 camera-only, and heavy fog improves by +0.23 on the
  enrichment-tagged slice. The +0.05 overall gain of seed 0 is within the seed
  spread, so it is not a reliable claim on its own.
- **Sync correction and trusted-only labels:** both differences (+0.017 for the
  uncorrected dataset, −0.037 for trusted-only labels) are inside the seed
  spread, so neither result is conclusive at this scale. Only 3 of 36 flights
  were sync-corrected.

## 4. Validation gate

| Candidate vs. baseline | mAP Δ (95% paired-bootstrap CI) | Decision |
|---|---|---|
| fusion vs. camera-only | +0.050 [+0.029, +0.070] | promote |
| camera-only vs. fusion | −0.050 [−0.070, −0.029]; night −0.085, heavy fog −0.230, vehicle −0.097, person −0.063 | **block** |
| fusion rerun vs. fusion | 0.000 [0.000, 0.000] | promote |
| fusion without sync correction vs. fusion | +0.017 [+0.005, +0.026] | promote |

The bootstrap resamples test frames, so it captures test-set sampling noise
but not training-run noise. Section 3 shows that training noise is the larger
of the two here: the "significant" +0.017 for the uncorrected dataset is well
inside the 0.057 seed spread. A promotion decision on a close call should use
several seeds per candidate (or a seed-noise floor in the gate config); the
gate already blocks the large, consistent regressions such as camera-only at
night.

## 5. AI tooling for data workflows

**Metadata enrichment** (taggers calibrated on the 6 validation flights, scored
on the other 30 flights, 4,741 frames): lighting tag accuracy 87.8%, fog tag
accuracy 79.0%; predicted object counts have a mean absolute error of 0.16
(vehicles), 0.44 (people) and 0.20 (buildings) per frame.

**Natural-language search** (offline parser; precision = share of returned
frames that truly match, by simulator truth):

| Query | Returned | Precision | Recall |
|---|---|---|---|
| night frames with at least 3 vehicles | 243 | 74% | 61% |
| foggy dusk frames with people | 221 | 75% | 40% |
| heavy fog with more than 2 vehicles | 140 | 58% | 53% |
| clear night frames with people | 368 | 54% | 31% |
| daytime frames with at least 2 buildings below 35 m | 18 | 33% | 27% |
| crowded frames | 13 | 92% | 80% |

Errors compound from the tagger and the detector's counts, which is why
queries that combine a hard condition (fog, night people) with counts are
weakest. Query-by-example: 57.8% of the 10 nearest frames (from other flights)
share the query's lighting and fog condition, vs. 16.6% for random frames.

**Label-error audit** (model trained on trusted labels, run over the 1,424
frames from the noisy vendor; 801 injected errors):

| Error type | Injected | Found |
|---|---|---|
| Shifted box | 324 | 283 (87%) |
| Wrong class | 122 | 81 (66%) |
| Missing box | 355 | 100 (28%) |

1,254 flags in total, 533 (43%) pointing at a real error, versus 17% of boxes
being wrong at random: reviewing the flags is about 2.5× more productive than
reviewing boxes at random. Missing boxes are the hardest case: a flag needs a
confident detection where no label exists.

**Auto-labelling** (labelled: 5 trusted-vendor flights, 790 frames; pool: 10
flights, 1,579 frames; thresholds calibrated on val for 95% precision):

| | test mAP |
|---|---|
| M0: trained on the labelled third | 0.522 |
| M1: labelled + pool (auto-accepted or human-reviewed labels) | 0.610 |
| Reference: human labels for all the same flights | 0.636 |

350 of 1,579 pool frames (22%) were auto-accepted. Their boxes were 98%
precise but recalled only 35% of the objects in those frames: a score-based
router can only see uncertainty about what the model detected, not objects it
missed entirely, so frames with confident misses look "easy". M1 still recovered 77% of the gap between M0 and the
reference while needing human labels for 78% of the pool. Catching confident
misses (e.g. a cross-check against LiDAR structure) is the next improvement.
