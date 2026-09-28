# 🎰 CasinoOps RL
## Reinforcement Learning for Casino Revenue Optimization

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Gymnasium](https://img.shields.io/badge/RL-Gymnasium-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Q--Learning-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📊 Project Architecture & Workflow

![CasinoOps RL Capstone Architecture](capstone.png)

---

## 📌 Project Overview

**CasinoOps RL** is a reinforcement learning project that explores how an intelligent agent can learn operational decision strategies from simulated casino gaming data.

I built a custom reinforcement learning environment using **Gymnasium** and implemented a **Q-learning agent from scratch** rather than relying on a prebuilt reinforcement learning model.

The agent observes machine operating conditions, selects one of three simulated operational actions, receives a reward, and gradually learns which actions produce stronger outcomes under different conditions.

The project demonstrates an end-to-end reinforcement learning workflow:

**Data Preparation → Environment Design → Q-Learning → Training → Evaluation → Experimentation → Business Interpretation**

> **Note:** This project uses synthetic data and a simulated reward system. It is a technical proof of concept and is not intended to provide real-world casino operational or gambling recommendations.

---

# 🎯 Business Problem

Casino gaming environments generate operational data related to machine activity, wagering behavior, payouts, maintenance, and revenue.

A traditional predictive model can estimate an outcome. Reinforcement learning addresses a different question:

> **Given the current operating conditions, what action should an agent take to maximize its expected reward?**

This project explores that question by creating a simulated decision-making environment in which an RL agent learns from repeated interactions.

The goal is to demonstrate how reinforcement learning can support **adaptive decision-making**, where different operating conditions can lead to different learned actions.

---

# 🗂️ Dataset

The project uses a synthetic casino revenue dataset containing approximately:

- **2,000 machine records**
- **45 simulated locations**
- Operational and revenue-related variables
- Machine activity and maintenance information

The dataset was created for educational experimentation.

### Key Features

| Feature | Description |
|---|---|
| `machine_id` | Unique machine identifier |
| `location_id` | Simulated location identifier |
| `store_name` | Fictional location name |
| `game_type` | Game category |
| `location_type` | Type of operating location |
| `days_active` | Number of active operating days |
| `avg_bet` | Average wager amount |
| `payout_rate` | Machine payout rate |
| `total_in` | Total amount wagered |
| `total_out` | Total amount paid out |
| `total_net` | Net gaming activity |
| `number_of_services` | Number of service events |
| `machine_swapouts` | Number of machine replacements |
| `monthly_revenue` | Monthly revenue |

Missing numerical values were handled using **median imputation** before the reinforcement learning environment was created.

---

# 🧠 Reinforcement Learning Design

A custom reinforcement learning environment was created using **Gymnasium**.

The environment defines:

- Observation space
- Action space
- State transitions
- Reward calculation
- Environment reset behavior
- Episode termination

The environment converts the casino dataset into a simulated decision-making problem that a reinforcement learning agent can interact with.

---

# 👁️ Observation Space

The RL agent observes five operational variables:

```text
avg_bet
payout_rate
days_active
number_of_services
machine_swapouts
```

Because tabular Q-learning requires discrete states, continuous variables were transformed using Scikit-learn's:

```python
KBinsDiscretizer
```

### State Discretization

| Feature | Bins |
|---|---:|
| Average Bet | 5 |
| Payout Rate | 5 |
| Days Active | 5 |
| Number of Services | 3 |
| Machine Swapouts | 2 |

This creates a manageable discrete state space for the Q-learning agent.

---

# 🎮 Action Space

The agent can select one of three simulated operational actions:

| Action | Description |
|---:|---|
| **0** | Maintain the current operating configuration |
| **1** | Apply an operational adjustment intended to improve performance |
| **2** | Apply an alternative optimization strategy |

The agent learns which actions provide stronger expected rewards under different operating conditions.

---

# 🤖 Q-Learning Agent

The Q-learning algorithm was implemented directly using **Python and NumPy**.

The Q-table has the following dimensions:

```text
(5, 5, 5, 3, 2, 3)
```

The first five dimensions represent the discretized state variables.

The final dimension represents the three available actions.

The Q-learning update follows:

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

The Q-table is updated as the agent interacts with the environment.

---

# 🔍 Exploration vs. Exploitation

The agent uses an **epsilon-greedy policy**.

Training begins with:

```text
epsilon = 1.0
```

This encourages exploration.

Epsilon gradually decreases until reaching:

```text
epsilon_min = 0.05
```

This shifts the agent toward using learned Q-values while maintaining a small amount of exploration.

---

# ⚙️ Training Configuration

The primary agent was trained using:

| Parameter | Value |
|---|---:|
| Training Episodes | 1,000 |
| Learning Rate (`alpha`) | 0.10 |
| Discount Factor (`gamma`) | 0.95 |
| Initial Epsilon | 1.00 |
| Minimum Epsilon | 0.05 |
| Epsilon Decay | 0.995 |

---

# 📈 Training Analysis

Training performance was evaluated using:

- Episode rewards
- Average reward
- Moving-average reward
- Q-value statistics
- Non-zero learned Q-values
- Agent action selections

A **50-episode moving average** was used to reduce short-term noise and make the overall reward behavior easier to interpret.

The notebook contains the complete training visualizations and results.

---

# 🧪 Agent Evaluation

After training, the learned policy was tested across **10 simulated machine states**.

### Action Distribution

| Action | Times Selected |
|---:|---:|
| Action 0 | 2 |
| Action 1 | 5 |
| Action 2 | 3 |

The agent used **all three available actions** during evaluation.

Action 1 was selected in **5 of the 10 evaluation cases**.

The variation in selected actions demonstrates that the learned policy is **state-dependent** rather than applying the same decision to every simulated machine condition.

---

# 🔬 Hyperparameter Experiment

To investigate the effect of learning rate, additional agents were trained using:

```text
α = 0.01
α = 0.10
α = 0.50
```

Other major training settings were kept consistent.

### Experimental Results

| Learning Rate | Average Reward | Maximum Q-Value | Non-Zero Q-Values |
|---:|---:|---:|---:|
| 0.01 | 11.951048 | 8.048067 | 128 |
| **0.10** | **12.210708** | 59.211095 | **134** |
| 0.50 | 11.797264 | 174.841844 | 125 |

For this experimental run, `α = 0.10` produced the highest average reward and the greatest number of non-zero learned Q-values.

The `0.50` learning rate generated substantially larger Q-values but did not improve average reward.

This experiment demonstrates why reinforcement learning hyperparameters should be tested rather than selected arbitrarily.

Because the environment contains randomized state selection and exploration, results can vary between runs.

---

# 💡 Key Findings

The project demonstrates several reinforcement learning concepts:

- The agent learned different decisions for different machine states.
- All three available actions were used during evaluation.
- Action 1 was selected most frequently in the 10-state evaluation.
- Epsilon decay transitioned the agent from exploration toward exploitation.
- State discretization allowed continuous operational features to be used with tabular Q-learning.
- Learning rate affected both Q-value magnitude and training behavior.
- `α = 0.10` produced the strongest average reward in the learning-rate experiment.

The primary takeaway is that the RL agent developed **state-dependent decision behavior** rather than selecting one universal action.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core development language |
| **JupyterLab** | Interactive development and analysis |
| **Pandas** | Data manipulation |
| **NumPy** | Q-table and numerical operations |
| **Scikit-learn** | State discretization |
| **Gymnasium** | Custom reinforcement learning environment |
| **Matplotlib** | Training and performance visualization |
| **Git** | Version control |
| **GitHub** | Project documentation and portfolio hosting |
| **Ubuntu Linux** | Development environment |

---

# 💼 Skills Demonstrated

This project demonstrates hands-on experience with:

### Data Science

- Data preparation
- Missing-value handling
- Feature selection
- Data transformation
- Experimental analysis
- Data visualization
- Result interpretation

### Machine Learning

- Reinforcement learning
- Q-learning
- Hyperparameter experimentation
- Model evaluation
- State discretization

### Reinforcement Learning

- Custom Gymnasium environments
- Observation spaces
- Action spaces
- Reward functions
- Q-tables
- Q-value updates
- Epsilon-greedy policies
- Exploration vs. exploitation
- Policy evaluation

### Software & Engineering

- Python development
- Linux
- Virtual environments
- JupyterLab
- Git
- GitHub
- Technical documentation
- Reproducible project organization

---

# 🏗️ Repository Structure

```text
casino-revenue-rl-optimization/
│
├── capstone.png
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

# 🚀 Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/christran85-cyber/casino-revenue-rl-optimization.git
```

### 2. Enter the Project Directory

```bash
cd casino-revenue-rl-optimization
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 4. Activate the Environment

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Start JupyterLab

```bash
jupyter lab
```

Open:

```text
notebooks/casino_revenue_rl_optimization.ipynb
```

Run the notebook from top to bottom.

---

# 📊 Project Results at a Glance

| Area | Result |
|---|---|
| Dataset | 2,000 synthetic machine records |
| Locations | 45 simulated locations |
| Environment | Custom Gymnasium environment |
| Algorithm | Q-Learning |
| State Features | 5 |
| Available Actions | 3 |
| Main Training Episodes | 1,000 |
| Final Exploration Rate | 0.05 |
| Evaluation States | 10 |
| Actions Used During Evaluation | All 3 |
| Most Selected Action | Action 1 |
| Learning Rates Tested | 0.01, 0.10, 0.50 |
| Highest Experimental Average Reward | 12.210708 at α = 0.10 |

---

# ⚠️ Limitations

This project is a technical proof of concept built in a simplified simulated environment.

Limitations include:

- Synthetic data
- Simulated reward logic
- Limited action space
- Limited observation space
- Short environment interactions
- Random exploration
- No real-world casino operational validation

The results demonstrate reinforcement learning concepts within the simulated environment and should not be interpreted as production casino recommendations.

---

# 🔮 Future Improvements

Future versions could include:

- Longer multi-step episodes
- More sophisticated reward shaping
- Additional operational actions
- Larger state spaces
- Multiple random-seed experiments
- Train/test environment separation
- Additional hyperparameter experiments
- Automated experiment tracking
- Deep reinforcement learning
- Comparison with DQN
- Comparison with PPO

These improvements would allow the project to progress from a tabular reinforcement learning proof of concept toward a more advanced RL experimentation platform.

---

# 🎯 Project Outcome

This project successfully demonstrates an end-to-end reinforcement learning workflow:

```text
Data
  ↓
State Representation
  ↓
Custom Gymnasium Environment
  ↓
Q-Learning Agent
  ↓
Training
  ↓
Policy Evaluation
  ↓
Hyperparameter Experimentation
  ↓
Business Interpretation
```

The project demonstrates the ability to take a business-oriented problem and translate it into a functioning reinforcement learning system using **Python, Gymnasium, NumPy, Pandas, Scikit-learn, and Matplotlib**.

---

# 👨‍💻 Author

**Chris Tran**

Building hands-on projects across:

- Data Science & Machine Learning
- Cloud Engineering
- IT Infrastructure
- Cybersecurity

---

## ✅ Project Status

**Complete**

**Module 2 Data Science Capstone — Reinforcement Learning**
