# 🎰 CasinoOps RL
## Casino Revenue Reinforcement Learning Optimization

A reinforcement learning project that explores how a Q-learning agent can learn operational decision strategies within a simulated casino gaming environment.

This project was developed as a **Module 2 Data Science Capstone** and demonstrates reinforcement learning concepts including custom environment design, state representation, reward-based learning, Q-learning, exploration versus exploitation, agent evaluation, visualization, and hyperparameter experimentation.

---

## 📌 Project Overview

Casino gaming operations involve many variables that may influence machine performance and revenue.

This project explores whether a reinforcement learning agent can learn different operational strategies based on simulated machine conditions.

The agent observes characteristics such as:

- Average bet
- Payout rate
- Days active
- Number of services
- Machine swapouts

The agent then selects one of three simulated operational actions and receives a reward based on the environment's revenue logic.

The objective is not to provide real-world casino recommendations, but to demonstrate how reinforcement learning can be applied to a simulated business optimization problem.

---

## 🎯 Project Goal

The goal of this project is to build and evaluate a reinforcement learning system capable of:

1. Representing casino machine operating conditions as states.
2. Selecting among multiple operational actions.
3. Receiving rewards from a simulated environment.
4. Learning action values through Q-learning.
5. Balancing exploration and exploitation.
6. Evaluating learned behavior across different machine conditions.
7. Measuring how hyperparameter changes affect learning performance.

---

## 🗂️ Dataset

The project uses a **synthetic casino game revenue dataset** containing:

- 2,000 records
- 45 simulated locations
- Machine-level operational characteristics
- Revenue and wagering information
- Service and maintenance information

Key variables include:

- `machine_id`
- `location_id`
- `store_name`
- `month`
- `game_type`
- `location_type`
- `distance_from_hq_miles`
- `days_active`
- `avg_bet`
- `payout_rate`
- `total_in`
- `total_out`
- `total_net`
- `number_of_services`
- `machine_swapouts`
- `monthly_revenue`

Missing numerical values were handled using median imputation before the reinforcement learning environment was created.

Because the dataset is synthetic, results should be interpreted as an educational proof of concept rather than evidence about actual casino operations.

---

## 🤖 Reinforcement Learning Environment

A custom environment was created using **Gymnasium**.

The environment models casino machine operating conditions and allows the agent to select among three possible operational actions.

### Observation Space

The state contains five features:

1. `avg_bet`
2. `payout_rate`
3. `days_active`
4. `number_of_services`
5. `machine_swapouts`

Continuous values were discretized using Scikit-learn's `KBinsDiscretizer` so they could be represented within a Q-table.

The resulting discrete state configuration uses:

- 5 bins for average bet
- 5 bins for payout rate
- 5 bins for days active
- 3 bins for number of services
- 2 bins for machine swapouts

---

## 🎮 Action Space

The agent can select one of three discrete actions:

| Action | Description |
|---|---|
| 0 | Maintain the current operating configuration |
| 1 | Apply an operational adjustment intended to improve performance |
| 2 | Apply an alternative optimization strategy |

The purpose of the action space is to simulate different operational decisions and allow the agent to learn which actions produce stronger rewards under different machine conditions.

---

## 🧠 Q-Learning Agent

The project implements a Q-learning agent using a multidimensional Q-table.

The Q-table has the following dimensions:

```text
(5, 5, 5, 3, 2, 3)
