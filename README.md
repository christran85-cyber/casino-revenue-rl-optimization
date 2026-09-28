# 🎰 CasinoOps RL
## Reinforcement Learning for Casino Revenue Optimization

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Gymnasium](https://img.shields.io/badge/RL-Gymnasium-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Q--Learning-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Project Overview

**CasinoOps RL** is a reinforcement learning project that explores how an intelligent agent can learn operational decision strategies from simulated casino gaming data.

I built a custom reinforcement learning environment using **Gymnasium** and implemented a **Q-learning agent from scratch** rather than relying on a prebuilt reinforcement learning model.

The agent observes machine operating conditions, selects one of three simulated operational actions, receives a reward, and gradually learns which actions produce stronger outcomes under different conditions.

The project demonstrates an end-to-end reinforcement learning workflow including:

- Data preparation
- Custom environment development
- State and action space design
- Reward design
- State discretization
- Q-learning
- Epsilon-greedy exploration
- Agent training
- Policy evaluation
- Reward visualization
- Hyperparameter experimentation
- Business interpretation

> **Note:** This project uses synthetic data and a simulated reward system. It is a technical proof of concept and is not intended to provide real-world casino operational or gambling recommendations.

---

# Business Problem

Casino gaming environments generate large amounts of operational data related to machine activity, betting behavior, payouts, maintenance, and revenue.

A traditional predictive model can estimate an outcome, but reinforcement learning addresses a different question:

> **Given the current operating conditions, what action should an agent take to maximize its expected reward?**

This project explores that question by creating a simulated decision-making environment in which an RL agent learns from repeated interactions.

The objective was to demonstrate how reinforcement learning could potentially support adaptive operational decision-making instead of applying the same strategy to every machine or location.

---

# Architecture

The project follows the following reinforcement learning workflow:

```text
Synthetic Casino Revenue Data
            │
            ▼
      Data Preparation
            │
            ▼
   Feature / State Selection
            │
            ▼
   State Discretization
            │
            ▼
 Custom Gymnasium Environment
            │
            ▼
      Q-Learning Agent
            │
            ▼
   Epsilon-Greedy Policy
            │
            ▼
      Agent Training
            │
            ▼
        Evaluation
            │
            ▼
 Reward & Behavior Analysis
            │
            ▼
 Hyperparameter Experiment
```

---

# Dataset

The project uses a synthetic casino game revenue dataset containing approximately **2,000 machine-level records** across **45 simulated locations**.

The dataset was created for educational experimentation and contains operational, wagering, maintenance, and revenue information.

## Example Features

| Feature | Description |
|---|---|
| `machine_id` | Unique machine identifier |
| `location_id` | Simulated location identifier |
| `store_name` | Fictional store/location name |
| `month` | Observation month |
| `game_type` | Game category |
| `location_type` | Type of operating location |
| `distance_from_hq_miles` | Distance from headquarters |
| `days_active` | Number of active operating days |
| `avg_bet` | Average wager amount |
| `payout_rate` | Machine payout rate |
| `total_in` | Total amount wagered |
| `total_out` | Total amount paid out |
| `total_net` | Net gaming activity |
| `number_of_services` | Number of maintenance/service events |
| `machine_swapouts` | Number of machine replacements |
| `monthly_revenue` | Monthly revenue |

Missing numerical values were handled using **median imputation** before the reinforcement learning environment was constructed.

---

# Reinforcement Learning Environment

I created a custom environment using **Gymnasium** to convert the casino dataset into a reinforcement learning problem.

The environment defines:

- Observation space
- Action space
- Environment reset behavior
- State transitions
- Reward calculation
- Episode termination

This allows the Q-learning agent to interact with the simulated casino environment through a standard reinforcement learning interface.

---

# Observation Space

The RL agent observes five operational variables:

```text
avg_bet
payout_rate
days_active
number_of_services
machine_swapouts
```

These features represent the operating state presented to the agent before it chooses an action.

Because traditional tabular Q-learning requires discrete states, continuous variables were transformed using Scikit-learn's:

```python
KBinsDiscretizer
```

The state configuration uses:

| Feature | Number of Bins |
|---|---:|
| Average Bet | 5 |
| Payout Rate | 5 |
| Days Active | 5 |
| Number of Services | 3 |
| Machine Swapouts | 2 |

This creates a manageable discrete state space for Q-learning.

---

# Action Space

The environment provides three possible actions:

| Action | Simulated Decision |
|---:|---|
| 0 | Maintain the current operating configuration |
| 1 | Apply an operational adjustment intended to improve performance |
| 2 | Apply an alternative optimization strategy |

The agent learns Q-values representing the expected value of taking each action from different operating states.

---

# Q-Learning Implementation

Instead of using a prebuilt RL agent, I implemented the Q-learning logic directly with Python and NumPy.

The Q-table dimensions are:

```text
(5, 5, 5, 3, 2, 3)
```

The first five dimensions represent the discretized observation variables.

The final dimension represents the three available actions.

The agent learns using the standard Q-learning update:

```text
Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:

```text
α = Learning rate
γ = Discount factor
r = Reward
s = Current state
s' = Next state
a = Selected action
```

Each interaction updates the Q-table so that the agent gradually develops preferences for particular actions under particular operating conditions.

---

# Exploration vs. Exploitation

The agent uses an **epsilon-greedy strategy**.

At the beginning of training:

```text
epsilon = 1.0
```

The agent therefore performs significant exploration.

After each training episode, epsilon decreases until reaching:

```text
epsilon_min = 0.05
```

This gradually shifts the agent from exploration toward exploitation of the learned Q-table while maintaining a small amount of exploration.

---

# Training Configuration

The primary agent was trained using:

| Parameter | Value |
|---|---:|
| Training Episodes | 1,000 |
| Learning Rate (`alpha`) | 0.10 |
| Discount Factor (`gamma`) | 0.95 |
| Initial Epsilon | 1.00 |
| Minimum Epsilon | 0.05 |
| Epsilon Decay | 0.995 |

The complete notebook can be found here:

**`notebooks/casino_revenue_rl_optimization.ipynb`**

---

# Training Analysis

Training performance was evaluated using:

- Episode rewards
- Average reward
- Moving-average reward
- Q-value statistics
- Non-zero learned Q-values
- Agent action selections

Two primary visualizations were generated.

## Q-Learning Training Rewards

The first visualization plots individual episode rewards.

Individual rewards vary because the environment presents different operating states and the epsilon-greedy policy intentionally explores different actions.

## 50-Episode Moving Average

A 50-episode moving average was also calculated.

This reduces short-term noise and makes broader reward behavior easier to analyze.

---

# Agent Evaluation

After training, the learned policy was tested across multiple simulated machine states.

During a 10-state evaluation, the agent selected:

| Action | Times Selected |
|---|---:|
| Action 0 | 2 |
| Action 1 | 5 |
| Action 2 | 3 |

**Action 1 was selected in 5 of the 10 evaluation cases.**

More importantly, the agent selected **all three actions** across the evaluation states.

This demonstrates that the learned policy was state-dependent rather than simply recommending the same action for every operating condition.

---

# Hyperparameter Experiment

To evaluate how the learning rate affected Q-learning behavior, I trained additional agents using three learning rates:

```text
0.01
0.10
0.50
```

All other major training settings were kept consistent.

## Results

| Learning Rate | Average Reward | Maximum Q-Value | Non-Zero Q-Values |
|---:|---:|---:|---:|
| 0.01 | 11.951048 | 8.048067 | 128 |
| 0.10 | **12.210708** | 59.211095 | **134** |
| 0.50 | 11.797264 | 174.841844 | 125 |

For this experimental run:

**`alpha = 0.10` produced the highest average reward and the greatest number of non-zero learned Q-values.**

The `0.01` learning rate updated values more gradually.

The `0.50` learning rate produced substantially larger Q-values but did not improve average reward.

This experiment illustrates why reinforcement learning hyperparameters should be evaluated rather than selected arbitrarily.

Because training includes random state selection and exploration, individual results can vary between runs.

---

# What the Agent Learned

The project demonstrates an important reinforcement learning concept:

> The best action can depend on the current state of the environment.

The trained agent did not choose the same action for every machine.

Instead, its decisions changed based on combinations of:

- Average wager
- Payout rate
- Operating days
- Maintenance frequency
- Machine swapouts

This is the main advantage being explored with reinforcement learning: **adaptive decision-making based on environmental state and learned rewards.**

---

# Technical Skills Demonstrated

This project demonstrates practical experience with:

### Python Development

- Functions
- Classes
- Loops
- Data structures
- NumPy arrays
- Pandas DataFrames

### Data Science

- Data inspection
- Missing-value handling
- Feature selection
- Data transformation
- Experimental analysis
- Result interpretation

### Machine Learning

- Reinforcement learning
- Q-learning
- State representation
- State discretization
- Hyperparameter tuning
- Model evaluation

### Reinforcement Learning

- Gymnasium environments
- Observation spaces
- Action spaces
- Reward functions
- Q-tables
- Bellman-style Q-value updates
- Epsilon-greedy policies
- Exploration vs. exploitation

### Visualization

- Matplotlib
- Training reward plots
- Moving averages
- Performance interpretation

### Engineering Workflow

- Linux development environment
- Python virtual environments
- JupyterLab
- Git version control
- GitHub
- Reproducible project structure
- Technical documentation

---

# Key Engineering Decisions

Several design decisions were made to keep the project understandable and reproducible.

### Custom Q-Learning Implementation

Q-learning was implemented directly rather than hiding the learning process behind a high-level library.

This makes the state-action-value updates visible and demonstrates understanding of the underlying reinforcement learning algorithm.

### State Discretization

Continuous operating features were converted into discrete bins so they could be represented efficiently in a multidimensional Q-table.

### Epsilon Decay

The agent begins by exploring heavily and gradually transitions toward its learned policy.

### Hyperparameter Testing

The learning rate was experimentally varied to observe its effect on reward and Q-table learning.

---

# Limitations

This project intentionally uses a simplified environment.

Important limitations include:

- Synthetic dataset
- Simulated reward structure
- Limited observation space
- Only three possible actions
- Short environment interactions
- No real casino operational validation
- Random exploration introduces run-to-run variability

The results therefore demonstrate **technical feasibility within a simulated environment**, not a production-ready decision system.

---

# Future Improvements

The project could be extended in several directions:

- Longer multi-step episodes
- More sophisticated reward shaping
- Additional operational actions
- Larger state spaces
- Multiple random-seed experiments
- Generalization testing
- More hyperparameter comparisons
- Train/test environment separation
- Additional reward-function experiments
- Comparison with other RL algorithms
- Deep reinforcement learning
- More realistic operational constraints

A future version could also compare tabular Q-learning against algorithms such as:

- Deep Q-Networks (DQN)
- Proximal Policy Optimization (PPO)

This would provide a useful comparison between traditional tabular reinforcement learning and modern deep RL approaches.

---

# Repository Structure

```text
casino-revenue-rl-optimization/
│
├── data/
│   └── casino_game_revenue_revised_with_store_names.csv
│
├── notebooks/
│   └── casino_revenue_rl_optimization.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/christran85-cyber/casino-revenue-rl-optimization.git
```

## 2. Enter the Project Directory

```bash
cd casino-revenue-rl-optimization
```

## 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

## 4. Activate the Environment

Linux/macOS:

```bash
source .venv/bin/activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Start JupyterLab

```bash
jupyter lab
```

Open:

```text
notebooks/casino_revenue_rl_optimization.ipynb
```

Then run the notebook from top to bottom.

---

# Project Results at a Glance

| Area | Result |
|---|---|
| Environment | Custom Gymnasium environment |
| RL Algorithm | Q-Learning |
| State Features | 5 |
| Available Actions | 3 |
| Main Training Episodes | 1,000 |
| Final Exploration Rate | 0.05 |
| Evaluation Tests | 10 |
| Most Selected Evaluation Action | Action 1 |
| Hyperparameters Tested | 3 learning rates |
| Strongest Learning Rate in Experiment | 0.10 |
| Best Experimental Average Reward | 12.210708 |

---

# Key Takeaway

CasinoOps RL demonstrates how a business problem can be transformed into a reinforcement learning environment and solved through an end-to-end machine learning workflow.

The project goes beyond simply training an agent. It includes:

**Problem Definition → Data Preparation → Environment Design → Q-Learning → Training → Evaluation → Visualization → Experimentation → Interpretation**

The result is a reproducible reinforcement learning proof of concept that demonstrates both **data science fundamentals and hands-on implementation skills**.

---

# Author

**Chris Tran**

Focused on building practical projects across:

- Data Science
- Machine Learning
- Cloud Engineering
- IT Infrastructure
- Cybersecurity

---

# Project Status

**Completed ✅**

Module 2 Data Science Capstone — Reinforcement Learning
