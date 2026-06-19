# EventHorizon: Market Regime Detection & Risk Analysis

## Academic Context

**Industrial Oriented Mini Project (IOMP)**

A quantitative machine learning architecture designed to identify latent macroeconomic states and dynamically adjust portfolio capital allocation to minimize maximum drawdowns during shock events.

---

## System Architecture

EventHorizon utilizes an unsupervised machine learning pipeline to process financial time-series data, bypassing the structural lag of traditional moving averages.

### Core Components

- **Dimensionality Reduction:** Principal Component Analysis (PCA) compresses fast-twitch OHLCV structural features.
- **Regime Detection:** Student-t Hidden Markov Models (HMM) utilize the Expectation-Maximization (EM) algorithm to classify distinct market environments (e.g., Bull, Bear, High-Volatility Shock) while robustly modeling fat-tailed extreme outliers.
- **Policy Routing:** A Temperature-Scaled Softmax engine translates HMM state probabilities into continuous, dynamically scaled portfolio weights.
- **Orchestration:** A Python/FastAPI backend serves the quantitative engine, while a Vite/React frontend dashboard provides out-of-sample analytics and visualization.

---

## Directory Structure

```text
EventHorizon/
├── src/                  # Core Python modules and ML logic
├── frontend/             # React 18 / Vite frontend application
├── api_models/           # Auto-generated .joblib model artifacts (Ignored by Git)
├── backtests/            # Auto-generated .csv performance data (Ignored by Git)
├── main.py               # ML Pipeline Orchestrator (Data -> PCA -> HMM -> Softmax)
├── api.py                # FastAPI router serving the trained models
├── requirements.txt      # Python dependencies
└── start_engine.bat      # Concurrency script to boot API and Frontend
```

---

## Tech Stack

### Backend Engineering
- Python 3.10+
- FastAPI
- Uvicorn

### Machine Learning & Quantitative Analytics
- Scikit-learn
- NumPy
- Pandas
- hmmlearn

### Frontend UI
- React 18
- Vite

### MLOps & Deployment
- Joblib (Model Serialization)
- Windows Batch Scripting (`.bat`)

---

## System Requirements

| Component | Requirement |
|------------|------------|
| Operating System | Windows 10 / 11 |
| Memory | 16 GB RAM minimum |
| Python | 3.10+ |
| Node.js | v18+ |

> **Note:** The expanding-window training architecture can consume significant memory during large-scale backtests and walk-forward validation.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/CodeWiz05/EventHorizon.git
cd EventHorizon
```

### 2. Python Environment Setup

Create and activate a virtual environment, then install all required dependencies.

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Frontend Setup

Install the React/Vite dependencies.

```bash
cd frontend

npm install

cd ..
```

---

## Execution Pipeline

### ⚠️ Critical Startup Sequence

The machine learning engine **must be trained before the API server is launched**.

The FastAPI layer depends on serialized `.joblib` artifacts generated during the training stage.

---

### Step 1 — Train the Engine & Generate Artifacts

Execute the primary orchestration pipeline from the project root directory.

This process performs:

1. Financial data ingestion
2. Feature engineering
3. PCA transformation
4. HMM training via EM optimization
5. Walk-forward validation
6. Softmax policy generation
7. Artifact serialization

Generated outputs:

- `api_models/*.joblib`
- `backtests/*.csv`

Run:

```bash
python main.py
```

---

### Step 2 — Launch the System

Once training completes successfully and the model artifacts have been generated, start the API server and frontend dashboard concurrently.

```bat
start_engine.bat
```

This script launches:

- FastAPI backend service
- React/Vite analytics dashboard

---

## Machine Learning Workflow

```text
Market Data (OHLCV)
          │
          ▼
Feature Engineering
          │
          ▼
Principal Component Analysis (PCA)
          │
          ▼
Student-t Hidden Markov Model (HMM)
          │
          ▼
State Probabilities
          │
          ▼
Temperature-Scaled Softmax
          │
          ▼
Dynamic Portfolio Weights
          │
          ▼
Backtesting & Risk Analytics
```

---

## Research Objectives

The EventHorizon framework is designed to:

- Detect latent market regimes before traditional indicators react.
- Quantify macroeconomic state transitions probabilistically.
- Dynamically reduce risk exposure during volatility shocks.
- Improve capital preservation during adverse market conditions.
- Provide interpretable state-based portfolio allocation signals.

---

## Development Notes & Limitations

### Lookahead Bias Mitigation

Sequence-based financial models are vulnerable to lookahead bias if future information leaks into the training process.

To address this issue, EventHorizon employs a:

- Strictly causal training methodology
- Expanding-window walk-forward validation framework
- Out-of-sample performance evaluation

This architecture more closely approximates real-world deployment conditions.

---

### Computational Overhead

The Gaussian HMM training procedure relies on the Expectation-Maximization (EM) algorithm implemented through `hmmlearn`.

Current limitations include:

- High CPU utilization during model fitting
- Long training times for large datasets
- Increased memory requirements during repeated walk-forward retraining

Potential future improvements:

- Vectorized feature generation
- Parallelized state estimation
- Compiled inference engines (Rust/C++)
- Online regime detection architectures
- GPU-accelerated probabilistic modeling

---

## Future Roadmap

### Quantitative Enhancements

- Bayesian Hidden Markov Models
- Regime-aware position sizing
- Dynamic volatility targeting
- Multi-asset allocation framework
- Reinforcement learning policy overlays

### Engineering Enhancements

- Docker containerization
- CI/CD pipelines
- Cloud deployment support
- Automated retraining schedules
- Experiment tracking and model registry integration

### Visualization Enhancements

- Interactive regime timelines
- Drawdown heatmaps
- State-transition matrices
- Real-time portfolio analytics
- Regime probability monitoring dashboards

---

## Disclaimer

EventHorizon is an academic research project developed for educational and quantitative experimentation purposes.

The system is **not financial advice**, does **not guarantee profitability**, and should not be used as the sole basis for investment decisions. All backtested results are historical simulations and may not reflect future market performance.