# Source-only anchor utility and experimental panel design

**Status:** post-hoc experimental-design extension. This output is isolated from all frozen HyperSpatial-MAP and HyperSpatial-MAP-T analyses.

## Questions answered

1. `anchor_utility.tsv` ranks each frozen anchor by leave-one-anchor-out source-CV utility.
2. `anchor_gene_utility.tsv` identifies the genes supported by each anchor without using coefficient magnitude.
3. `anchor_redundancy.tsv` reports ordered conditional utility $U_{a|b}$; lower values indicate stronger substitution/redundancy.
4. `panel_design_8.tsv`, `panel_design_16.tsv`, and `panel_design_32.tsv` give new source-only experimental recommendations.

## Main descriptive findings

- `50p_E1_to_E2`: highest frozen-panel LOAO utility `cxcr4a` (median Δ residual Pearson 0.0031; 77.0% of evaluated genes helped).
- `50p_E2_to_E1`: highest frozen-panel LOAO utility `gata5` (median Δ residual Pearson 0.0021; 70.4% of evaluated genes helped).
- `6s_E1_to_E2`: highest frozen-panel LOAO utility `fn1b` (median Δ residual Pearson 0.0029; 74.9% of evaluated genes helped).
- `6s_E2_to_E1`: highest frozen-panel LOAO utility `rrbp1b` (median Δ residual Pearson 0.0025; 69.7% of evaluated genes helped).
- `75p_E1_to_E2`: highest frozen-panel LOAO utility `cdh6` (median Δ residual Pearson 0.0054; 73.1% of evaluated genes helped).
- `75p_E2_to_E1`: highest frozen-panel LOAO utility `rrbp1b` (median Δ residual Pearson 0.0020; 66.4% of evaluated genes helped).

## Budget-16 experimental recommendations

- `50p_E1_to_E2` budget 16: krt4, ism1, efnb2a, cdkn1ca, dnmt3bb.2, foxd5, tmed2, fbln2, zgc:174154, casz1, pcdh8, id3, baz1b, kdrl, tmed10, cdx4.
- `50p_E2_to_E1` budget 16: krt4, gata5, oclna, si:ch211-113a14.12, snai2, mdm2, six7, cmtm6, nanos3, tcf7l2, hic1l, atoh1c, zgc:175088, tph1b, wnt8a, C7H16orf87.
- `6s_E1_to_E2` budget 16: six4a, prmt1, rrbp1b, nsd2, tfap2a, sox11a, cenpf, cdh6, nid2a, foxd5, lbx2, zeb1b, olig4, cst3, efnb2a, tbx16l.
- `6s_E2_to_E1` budget 16: snai1a, tfap2a, tuba1a, fn1b, rrbp1b, cenpf, akap12b, cdh6, nsd2, fn1a, f11r.1, prdm1a, tbx16l, kirrel3l, zic3, ephb3a.
- `75p_E1_to_E2` budget 16: znf185, snai1a, xbp1, tuba8l2, bmp4, sox11b, lmo4a, irx1b, cdh2, fezf2, foxn2b, sox5, cxcr4a, fzd7b, dynll1, foxi1.
- `75p_E2_to_E1` budget 16: lamb1a, sox11a, klf2b, lamp2, id3, otx2b, zic2a, foxd5, sox2, irf2a, foxi1, ets2, her9, rrbp1b, perp, nkx2.5.

These budget-16 recommendations are experimental-design results only. They do not replace, revise, or reinterpret the frozen 16-gene panels used by the manuscript analyses. No claim is made that 16 genes is universally optimal.

## Methods

Each embryo is analyzed only in its source role. Counts are normalized exactly as in the frozen source runner. Coordinates use the frozen loader choice (`global_sphere` for 50%/75%; raw `spatial` for 6-somite), median centering, and 99th-percentile radial scaling. Eight source spatial pseudo-holdouts are the frozen coordinate octants. Registered 12-neighbour IDW is reproduced only within a source embryo to define pseudo-holdout residuals; no cross-embryo registration is run.

For a frozen panel $A$, `PerfCV_g(A)` is eight-fold source spatial-CV residual Pearson from a centered ridge predictor with the setting's already-frozen ridge, gain, and rank cap. Leave-one-anchor-out utility is `PerfCV_g(A) - PerfCV_g(A\{a})`. The MSE column is oriented so positive means that retaining the anchor lowers MSE. Evaluation genes exclude the complete frozen 16-anchor panel.

Conditional utility is `PerfCV(A\{b}) - PerfCV(A\{a,b})`. Low or negative conditional utility means anchor `a` adds little once `b` is absent and is therefore more substitutable/redundant; it is not coefficient importance.

Panel design uses strict nested source CV. Each outer octant is untouched during panel selection. The other seven octants provide inner spatial folds. Greedy additions maximize the conditional marginal inner-CV residual Pearson of a stagewise centered-ridge residual decoder. Because every candidate is evaluated after conditioning on the already-selected panel, redundant candidates receive little marginal utility. Frozen eligibility/exclusion rules are retained. Coverage is the frozen source criterion `mean_g max_a |r_source(g,a)|²`; redundancy is mean pairwise squared source correlation. After unbiased outer-fold evaluation, the final recommendation is selected using all source cells with the same eight spatial folds. No target expression is accessed.

The embryo is the biological unit. Cells, genes, gene-by-anchor records, and spatial folds are technical evaluation units. Reciprocal source settings are not independent biological replicates.

## Files

- `anchor_utility.tsv`: anchor-level LOAO summaries.
- `anchor_gene_utility.tsv`: per-anchor/per-gene utilities.
- `anchor_redundancy.tsv`: ordered conditional-utility matrix in long form.
- `panel_design_{8,16,32}.tsv`: nested source-only recommended panels.
- `panel_budget_performance.tsv`: outer-fold and aggregate source-CV performance.
- `panel_design_summary.pdf/png`: descriptive extension summary.
- `INPUT_HASHES.tsv`, `PROVENANCE.md`, and `00_audit/`: isolation and integrity records.
