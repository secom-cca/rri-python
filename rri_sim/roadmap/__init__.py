"""Climate-adaptation roadmaps (pathways) under deep uncertainty: explainability and updateability.

    config      : settings (measures and their delays, participation, world factors, criteria regimes, policies)
    physics     : PHYSICAL layer - the RRI-trained surrogate of an exploration run (rri_sim.rdm.explore),
                  measure costs (costs.py), RRI re-evaluation of selected states
    social      : SOCIAL / INSTITUTIONAL layer - stakeholder engagement, consensus time and failure,
                  adoption ceilings, coordination cost
    world       : one possible future (climate, land use, measure effects, delays, social parameters,
                  criteria regime) from factors in the unit cube; observations of it
    sim         : implementation state (decided / consensus / construction / diffusion / maturation,
                  budget, maintenance) and the simulation of a roadmap controller in a world
    planner     : roadmaps, their re-optimisation from an inherited state (GA on the surrogate), and the
                  policies P0 (fixed), P1 (DAPP), P2 (+ model updating), P3 (+ criteria updating)
    metrics     : regret, updateability, future option coverage, lock-in, explainability (Morris, Sobol, PRIM)
    experiments : early decision / delays, portfolio complementarity, stakeholder participation
    run         : the pipeline -> data/processed/roadmap/<name>/results.json (GUI /roadmap)

The physical layer (RRI, through its surrogate) and the social / institutional
layer are kept separate: sim.py turns the implementation state into measure
levels, and only physics.py maps levels and climate to flood outcomes.
See docs/13_roadmap.md.
"""
