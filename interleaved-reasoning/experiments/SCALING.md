# Scaling in N

Task A, MLP stages, d=96, 2500 steps, N ∈ {4, 8, 12, 24, 48}
(`A_scale_n4 … A_scale_n48`).

What we expect to observe (and what the JSON reports):

- **Criterion 1 (non-degeneracy) and criterion 3 (parity)** are properties of
  the *mechanism*, not of the depth: they must pass at every N. If a large
  N collapses, the non-degeneracy floor is interacting with the batch
  statistic in a depth-dependent way — that would be a finding.
- **Mean emissions** should grow with N (more stages = more opportunities to
  anchor) and saturate below N (the price caps it).
- **Accuracy** should be U-shaped or flat in N: too few stages cannot do the
  work; too many stages multiply the channel-noise exposure of the latent
  path, which the controller answers with more emissions — accuracy should
  hold because emissions counteract the extra drift.
- **Corr(measured, emits)** should stay > 0.3: the adaptive signal is about
  *where in the sequence* to anchor, which exists at every depth.

No baseline comparison (criterion 4) is attached to the scaling rows: the
baselines are the N=12 pair of the main runs, and cross-N CE comparison is
not a controlled comparison. C1/C3/correlation are the scaling metrics.
