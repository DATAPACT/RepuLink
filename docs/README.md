# RepuLink

Powered by

[![UoSLOGO](./images/UniSouthampton.png)](https://dips.soton.ac.uk/#home)



## **General Description**

**RepuLink** is a network-based reputation system that combines traditional forward propagation with a novel backward propagation mechanism to adjust reputations based on both interactions and endorsements. The algorithm leverages endorsement penalty and reward signals to provide a more robust, calibrated, and explainable reputation evaluation in decentralized networks.

![Repulink](../figure/Repulink.png)


## **Commercial Information**

Table with the organisation, license nature (Open Source, Commercial ... ) and the license. Replace with the values of your module.

| Organisation (s) | License Nature | License |
| ---------------  | -------------- | ------- |
| University of Southampton  | Open Source | MIT Licence  |


## **Top Features**

In decentralized networks, establishing trust between unknown or partially trusted nodes is a key challenge. **Repulink** introduces a two-layer reputation model that:

1. **Integrates multiple types of feedback:**  
   - **Interaction data:** Captures individual feedback (positive/negative) between nodes.
   - **Endorsement data:** Captures social trust signals through endorsements.

2. **Applies backward propagation mechanisms:**  
   - **Penalty Propagation (BEPP):** Propagates negative feedback from misbehaving nodes back to endorsers.
   - **Reward Propagation (BERP):** Rewards endorsers when the nodes they endorse perform well.

3. **Calibrates final reputations:**  
   - Combines forward reputation computed via power iteration with backward adjustments.
   - Maintains normalized reputation distributions to reflect network performance.

## **How To Install**

All configurable parameters are stored in `config.py`, including:

- `EPSILON`: Small constant to prevent division by zero.
- `ALPHA`: Weight between local trust and endorsement contribution.
- `LAMBDA`: Reward sensitivity coefficient.
- `BETA`: Penalty sensitivity coefficient.
- `GAMMA`: Penalty (reward) discount factor.
- `MAX_ITER`: Maximum iterations for convergence.
- `CONVERGENCE_THRESHOLD`: Convergence threshold for the reputation vector.

Feel free to adjust these parameters as needed for experimentation and simulation tuning.

## **How To Use**

The simulation runs via the main driver script `main.py`, which loads processed data, instantiates managers, performs reputation computation, and records node histories.

To run the simulation and save output to a file, you can either use command-line redirection or configure logging in the script.

Example using command-line redirection:

```bash
python3 repulink_example_full.py 
```
Check the results saved under the folder `repulink_multislot_core_modules`.

## Expected KPIs

These KPI relate to the Policy Service suite of tools, which include the Policy Editor, the Policy Enginge and the Ontology Service.

| What | How | Values |
| --- | --- | --- |
| 1) Enforcing trust accountability through endorsement chain 2) Efficient reputation updating with incentive and accountability mechanisms	| 1) Testing with synthetic data (user profiles with different behaviours) to check if the endorsers of bad-performed users will be penalised 2) Efficiency of the RepuLink algorithm (comparison of reputation updating time with/without incentive/accountability mechanisms)	| 1) Set a threshold for penalty, only the penalty lower than the threshold, the backward propagation will stop. 2) Achieve similar efficiency with other similar algorithms for trust and reputation calculation. |

