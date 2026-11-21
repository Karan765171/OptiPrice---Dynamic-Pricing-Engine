# OptiPrice

**OptiPrice** is an autonomous dynamic pricing engine simulator in Python that optimizes sequential pricing decisions under uncertainty while enforcing mathematical volatility jump guardrails to prevent erratic price swings.

---

## Overview

Dynamic pricing engines often face a tension between chasing immediate reward and gathering enough information about unfamiliar price points—while aggressive price changes can trigger customer churn, instability, or unrealistic jumps in real-world operations.

**OptiPrice** simulates dynamic price selection over discrete time steps using multi-armed bandit policies combined with explicit volatility guardrails, allowing you to compare how different exploration–exploitation strategies behave under stability constraints.

---

## Key Features

* **Multi-Armed Bandit Pricing Policies:** Evaluates candidate price arms over iterative simulation rounds using:
  * **Epsilon-Greedy:** Balances random exploration across price arms with exploitation of the highest estimated reward arm.
  * **Upper Confidence Bound (UCB):** Selects price arms optimistically based on both estimated value and uncertainty bounds that shrink as an arm is sampled more often.
  * **Thompson Sampling:** Samples from posterior reward distributions to naturally shift from broad exploration early on to focused exploitation as evidence accumulates.
* **Mathematical Volatility Jump Guardrails:** Constrains abrupt step-to-step price changes so proposed price updates remain within safe bounds rather than jumping wildly between distant price tiers.
* **Simulation & Visual Analysis:** Generates structured simulation logs and interactive charts using **NumPy**, **Pandas**, and **Plotly** to inspect reward accumulation, arm selection frequency, and price stability.

---

## How It Works

1. **Define Candidate Price Arms & Simulation Parameters:** Configure the set of discrete price points, simulation horizon (number of steps/rounds), bandit hyperparameters, and volatility guardrail thresholds.
2. **Select a Price Arm at Each Step:** The active bandit policy (**Epsilon-Greedy**, **UCB**, or **Thompson Sampling**) scores or samples candidate prices based on historical observations and uncertainty.
3. **Apply Volatility Jump Guardrails:** Before locking in the candidate price, the guardrail layer checks the change relative to the current/previous price and clamps or restricts excessive jumps according to the configured stability rule.
4. **Observe Feedback & Update State:** Simulated demand/reward feedback is recorded for the executed price, updating the policy's internal estimates and appending step metrics to a **Pandas** dataset.
5. **Analyze Results:** Visualize price trajectories, cumulative/average rewards, exploration vs. exploitation behavior, and guardrail interventions using **Plotly**.

---

## Tech Stack

* **Language:** Python
* **Core Scientific & Data Libraries:** NumPy, Pandas
* **Visualization:** Plotly
* **Optimization & Control Concepts:** Multi-Armed Bandits (Epsilon-Greedy, UCB, Thompson Sampling), Volatility Jump Guardrails

---

