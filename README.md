# QuantFlux


<!-- 
This respository is a quantitative research tool focused on understanding the relationship between stock-price dynamics, option values, machine-learning predictions and statistical inference.

The goal is not only to build models that generate accurate predictions, but also to understand why they work, when they fail, and whether their outputs are statistically meaningful.

In this repository we combine historical stock-price analysis, simulated stock-price paths, option pricing and option sensitivities, machine-learning models for financial prediction, statistical hypothesis testing and uncertainty estimation, model interpretation and comparison, and visualization of financial and statistical results.

## Project Goals

This repository explores several connected questions:

1. How well do standard stochastic models reproduce observed stock-price behavior?
2. How do assumptions about returns and volatility affect simulated price paths?
3. How sensitive are option prices to different market conditions and modeling assumptions?
4. Can machine-learning models improve predictions of stock returns, volatility, or option values?
5. Are improvements in predictive performance statistically meaningful?
6. Which features drive model predictions?
7. How stable are model conclusions across time periods and changing market regimes?

Rather than treating prediction accuracy as the final result, the project emphasizes statistical interpretation, robustness, and model behavior.

## Repository Structure 

### TENTATIVELY

```text
quantflux/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_market_data_exploration.ipynb
│   ├── 02_stock_path_simulation.ipynb
│   ├── 03_option_pricing.ipynb
│   ├── 04_volatility_analysis.ipynb
│   ├── 05_machine_learning.ipynb
│   └── 06_statistical_interpretation.ipynb
│
├── src/
│   ├── data/
│   ├── simulation/
│   ├── options/
│   ├── features/
│   ├── models/
│   ├── statistics/
│   └── visualization/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── model_outputs/
│
├── tests/
│
├── requirements.txt
└── README.md
```

### Currently

```text
quantflux/
│
├── .gitignore
├── requirements.txt
└── README.md
```

The exact structure will evolve as the project grows.
---

## 1. Market Data and Exploratory Analysis

The first part of the project studies historical price behavior.

Potential analyses include:

- Price and return distributions
- Log returns
- Rolling volatility
- Autocorrelation
- Drawdowns
- Correlation across assets
- Tail behavior
- Distributional assumptions
- Volatility clustering
- Market-regime comparisons

The goal is to identify which statistical properties of real financial data should be reproduced by later simulation and modeling steps.

---

## 2. Stock-Price Path Simulation

The repository will compare observed stock behavior with simulated paths generated from common stochastic models.

Initial models may include:

### Geometric Brownian Motion

$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$

GBM provides a useful baseline for studying the consequences of assuming constant drift, constant volatility, and normally distributed log returns.

### Historical Bootstrap

Historical returns can also be resampled to generate paths without imposing a normal return distribution.

### Time-Varying Volatility Models

Extensions may include models such as:

- GARCH
- Regime-dependent volatility
- Stochastic volatility

Simulated paths can then be compared using quantities such as:

- Return distributions
- Realized volatility
- Drawdowns
- Extreme moves
- Path-dependent statistics

---

## 3. Option Pricing

The project connects stock-price dynamics to derivative valuation.

Initial methods may include:

### Black-Scholes

European option prices can be computed using the Black-Scholes model and used as a theoretical benchmark.

### Monte Carlo Pricing

Simulated stock paths can be used to estimate option values numerically.

This allows direct comparison between:

- Analytical prices
- Monte Carlo estimates
- Market prices
- Machine-learning predictions

Additional analyses may examine:

- Calls and puts
- Moneyness
- Time to expiration
- Implied volatility
- Greeks
- Volatility smiles and skews
- Sensitivity to model assumptions

---

## 4. Machine Learning

Machine-learning models will be used to study financial prediction problems such as:

- Future returns
- Future volatility
- Option prices
- Implied volatility
- Directional price movement

Candidate models include:

- Linear regression
- Logistic regression
- Regularized regression
- Random forests
- Gradient-boosted trees
- Neural networks

The focus is not simply on using increasingly complex models. Every model will be compared against simple financial and statistical baselines.

### Example Features

Potential predictors include:

- Lagged returns
- Rolling volatility
- Moving averages
- Momentum indicators
- Volume
- Drawdown statistics
- Market index returns
- Option moneyness
- Time to expiration
- Risk-free rates
- Historical volatility
- Implied volatility

---

## 5. Statistical Analysis

Model performance will be evaluated using more than a single predictive metric.

Depending on the problem, analyses may include:

- Confidence intervals
- Bootstrap uncertainty
- Hypothesis tests
- Residual diagnostics
- Out-of-sample error
- Time-series cross-validation
- Distribution comparisons
- Correlation analysis
- Calibration
- Statistical significance of model improvements

For example, if two models have different test errors, the project will investigate whether that difference is large relative to the statistical uncertainty of the evaluation.

---

## 6. Statistical Interpretation

A central goal of the repository is to distinguish between:

> **A model producing a better metric**

and

> **Evidence that the model has learned a stable and economically meaningful relationship**

Model interpretation may include:

- Feature importance
- Permutation importance
- Partial dependence
- SHAP values
- Residual analysis
- Prediction distributions
- Error decomposition
- Performance across volatility regimes
- Performance across market periods

This allows the project to investigate questions such as:

- Does a model depend heavily on recent volatility?
- Does performance disappear during high-volatility periods?
- Are improvements concentrated in only a small number of observations?
- Do predictions remain stable when the data distribution changes?
- Are statistically significant improvements economically meaningful?

---

## 7. Model Validation

Financial machine learning is especially vulnerable to overfitting and data leakage.

The project therefore emphasizes validation practices such as:

- Chronological train-validation-test splits
- Walk-forward validation
- Time-series cross-validation
- Feature construction using only past information
- Out-of-sample evaluation
- Comparison against naive baselines

Random train-test splitting will generally be avoided when it would allow future market information to influence past predictions.

---

## Example Research Workflow

A typical experiment in this repository may follow the sequence:

```text
Historical Market Data
        ↓
Exploratory Statistical Analysis
        ↓
Feature Engineering
        ↓
Baseline Financial Model
        ↓
Machine-Learning Model
        ↓
Out-of-Sample Evaluation
        ↓
Statistical Comparison
        ↓
Model Interpretation
        ↓
Economic Interpretation
```

This structure keeps prediction, statistical evidence, and financial interpretation connected.

---

## Example Research Questions

Some experiments planned for the repository include:

- How different are empirical stock-return distributions from the assumptions of GBM?
- Does a GARCH volatility model generate more realistic stock paths than constant-volatility GBM?
- How much Monte Carlo uncertainty is present in simulated option prices?
- Can machine learning reproduce Black-Scholes prices from option inputs?
- Can machine learning improve option-price predictions when market prices deviate from simplified theoretical assumptions?
- Which variables contribute most strongly to predicted implied volatility?
- Does additional model complexity produce statistically significant out-of-sample improvements?
- How does model performance change between low- and high-volatility market regimes?
- How stable are feature importances over time?

---

## Evaluation Metrics

Depending on the problem, model performance may be evaluated with:

### Regression

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- $R^2$
- Relative pricing error

### Classification

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

### Financial Evaluation

Where appropriate, statistical metrics may be supplemented with quantities such as:

- Directional accuracy
- Volatility forecast error
- Option-pricing error
- Return distributions
- Drawdown behavior
- Risk-adjusted performance

Predictive performance alone will not automatically be interpreted as evidence of a profitable trading strategy.

---

## Technologies

The project is primarily implemented in Python.

Core libraries may include:

```text
NumPy
Pandas
SciPy
Matplotlib
scikit-learn
statsmodels
PyTorch
yfinance
tensorflow
keras
```

Additional packages may be introduced as the project develops.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/luisfelipegz/quantflux.git
cd quantflux
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Current Development Plan

### Phase 1
- Historical stock-data pipeline
- Exploratory return and volatility analysis
- GBM stock-path simulation
- Monte Carlo visualization

### Phase 2
- Black-Scholes implementation
- Monte Carlo option pricing
- Greeks and implied-volatility analysis
- Comparison of simulated and theoretical prices

### Phase 3
- Machine-learning baselines
- Feature engineering
- Time-series validation
- Model comparison

### Phase 4
- Statistical uncertainty analysis
- Bootstrap comparisons
- Feature interpretation
- Regime-dependent performance analysis

### Phase 5
- More realistic volatility models
- Extended option datasets
- Robustness studies
- End-to-end research case studies

---

## Reproducibility

Where possible, experiments will include:

- Fixed random seeds
- Explicit model parameters
- Saved configuration settings
- Reusable data-processing functions
- Separate training and evaluation code
- Reproducible figures and tables

The objective is for results to be easy to verify and extend.

-->

---

## Disclaimer

This repository is intended for **research and educational purposes only**.

Nothing in this project should be interpreted as financial advice, an investment recommendation, or evidence that a particular strategy will remain profitable in future markets.

---

## Author

**Luis Felipe Gutierrez**

Physics PhD transitioning quantitative research, statistical modeling, and machine learning methods toward financial applications.

GitHub: [github.com/luisfelipegz](https://github.com/luisfelipegz)
