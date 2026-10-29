"""Dynamic Adaptive Policy Pathways (DAPP) on top of the RRI model and the RDM tools.

    config : settings (time steps, climate paths, measure lead times, strategies, ATP thresholds)
    model  : simulate one strategy under one climate path, decade by decade
    run    : all strategies x climate paths -> ATPs, scores, regret, robustness, pathways map data

Each decade is represented by one flood event (the basin's event, scaled by the
climate of the decade's mid-year) with a fixed annual exceedance probability;
see docs/11_dapp.md for this approximation and its limits.
"""
