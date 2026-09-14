# Source-only panel design

`design_panel(reference, budget=16)` constructs a source/stage-specific measurement panel using nested spatial cross-validation. Eligibility, coverage, greedy marginal recoverability, leave-one-anchor-out utility and conditional utility are computed from the reference only.

Budgets 8, 16 and 32 reproduce the post-hoc experimental-design analysis convention; custom positive budgets are supported as a tool convenience. A designed budget-16 panel does not replace the frozen manuscript panels, and neither the gene set nor panel size is claimed to be universally optimal.

Lower conditional utility indicates greater substitutability. Genes and source-CV records are technical evaluation units; the source specimen remains the biological unit.

