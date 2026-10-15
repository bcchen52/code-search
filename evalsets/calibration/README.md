# Judge calibration

Judged metrics are trusted only after the judge agrees with human labels.
Hand-label 40 answers from one dev run, stratified by category, in
`labels.yaml`:

```yaml
run_id: 2026-10-14T143207-3f2a9c1e
labels:
  tiger-regalloc-003: 2     # 2 correct and complete, 1 partial, 0 wrong
  tcp-maxconn-011: 0
```

Label without looking at the judge's verdicts, then run
`cqa eval calibrate --run-id <run>`. It reports raw agreement and Cohen's
kappa, overall and per category. Below 0.6, revise the rubric as a new prompt
version in `cqa/eval/prompts/` before trusting judged metrics.
