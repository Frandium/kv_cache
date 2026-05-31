# Protocol For Approval: 2x2 Top-1 Router Lock-In

Goal: decide whether top-1 collapse is caused by early router geometry.
Decision question: does step 0/10 assignment determine final route separation?
Tested hypothesis: same-expert initial assignment causes lock-in and starvation.
Rival explanation: collapse appears later from optimization drift or task shortcut.
Data: fixed orthogonal states $h_A=e_1,h_B=e_2$ with A/B next-token targets.
Model: 2 linear experts, linear top-1 router, linear LM head.
Routing / algorithm: selected top-1 expert only; router gradient through selected gate.
Loss / objective: next-token CE.
Conditions: baseline random router, success control, no-router-bias diagnostic.
Checkpoints / seeds: steps 0,10,50,100,200,300; seeds 0-2 for trajectory, 0-9 for bias comparison.
Primary metric: NMI and top-1 assignment trajectory.
Secondary metrics: selected gate, update norm, counterfactual CE, token accuracy.
Known good case: initial A/B split should stay split.
Known bad case: initial A/B same expert should collapse.
Known confusing case: token accuracy 1.0 with NMI 0.
Success: trajectory distinguishes lock-in from later drift.
Failure: assignments change late without early lock-in evidence.
Insufficient evidence: missing checkpoint, seed, or counterfactual CE.
What this cannot claim: full LM behavior, attention behavior, utility-aligned specialization.
User approval checklist: hypothesis / primary metric / boundary / next decision.
