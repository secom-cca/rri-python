"""Robust Decision Making (RDM) for basin flood management on top of the RRI model.

    config      : analysis settings (JSON; saved with every run for reproducibility)
    candidates  : candidate measures from site-suitability conditions -> decision variables
    evaluate    : policy x scenario -> metrics, through the GUI engine (cached on disk)
    nsga2       : constrained NSGA-II for integer decision variables
    uncertainty : scenario design (climate x LHS), post-hoc factors, surrogate, Sobol indices
    robustness  : stakeholder utilities, regret, minimax-regret recommendation
    run         : the pipeline (optimise -> stress test -> sensitivity -> robustness)

The RRI physics is used unchanged: measures are GUI edits, scenarios are the
existing climate / parameter inputs.  See docs/09_rdm.md.
"""
