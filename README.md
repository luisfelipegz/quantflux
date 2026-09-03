# QuantFlux

[comment]: # This respository is a quantitative research tool focused on understanding the relationship between stock-price dynamics, option values, machine-learning predictions and statistical inference.
[comment]: # 
[comment]: # The goal is not only to build models that generate accurate predictions, but also to understand why they work, when they fail, and whether their outputs are statistically meaningful.
[comment]: # 
[comment]: # In this repository we combine historical stock-price analysis, simulated stock-price paths, option pricing and option sensitivities, machine-learning models for financial prediction, statistical hypothesis testing and uncertainty estimation, model interpretation and comparison, and visualization of financial and statistical results.
[comment]: # 
[comment]: # ## Project Goals
[comment]: # 
[comment]: # This repository explores several connected questions:
[comment]: # 
[comment]: # 1. How well do standard stochastic models reproduce observed stock-price behavior?
[comment]: # 2. How do assumptions about returns and volatility affect simulated price paths?
[comment]: # 3. How sensitive are option prices to different market conditions and modeling assumptions?
[comment]: # 4. Can machine-learning models improve predictions of stock returns, volatility, or option values?
[comment]: # 5. Are improvements in predictive performance statistically meaningful?
[comment]: # 6. Which features drive model predictions?
[comment]: # 7. How stable are model conclusions across time periods and changing market regimes?
[comment]: # 
[comment]: # Rather than treating prediction accuracy as the final result, the project emphasizes statistical interpretation, robustness, and model behavior.
[comment]: # 
[comment]: # ## Repository Structure 
[comment]: # 
[comment]: # ### TENTATIVELY
[comment]: # 
[comment]: # ```text
[comment]: # quantflux/
[comment]: # │
[comment]: # ├── data/
[comment]: # │   ├── raw/
[comment]: # │   └── processed/
[comment]: # │
[comment]: # ├── notebooks/
[comment]: # │   ├── 01_market_data_exploration.ipynb
[comment]: # │   ├── 02_stock_path_simulation.ipynb
[comment]: # │   ├── 03_option_pricing.ipynb
[comment]: # │   ├── 04_volatility_analysis.ipynb
[comment]: # │   ├── 05_machine_learning.ipynb
[comment]: # │   └── 06_statistical_interpretation.ipynb
[comment]: # │
[comment]: # ├── src/
[comment]: # │   ├── data/
[comment]: # │   ├── simulation/
[comment]: # │   ├── options/
[comment]: # │   ├── features/
[comment]: # │   ├── models/
[comment]: # │   ├── statistics/
[comment]: # │   └── visualization/
[comment]: # │
[comment]: # ├── results/
[comment]: # │   ├── figures/
[comment]: # │   ├── tables/
[comment]: # │   └── model_outputs/
[comment]: # │
[comment]: # ├── tests/
[comment]: # │
[comment]: # ├── requirements.txt
[comment]: # └── README.md
[comment]: # ```
[comment]: # 
[comment]: # ### Currently
[comment]: # 
[comment]: # ```text
[comment]: # quantflux/
[comment]: # │
[comment]: # ├── .gitignore
[comment]: # ├── requirements.txt
[comment]: # └── README.md
[comment]: # ```
[comment]: # 
[comment]: # The exact structure will evolve as the project grows.
[comment]: # ---
[comment]: # 
[comment]: # ## 1. Market Data and Exploratory Analysis
[comment]: # 
[comment]: # The first part of the project studies historical price behavior.
[comment]: # 
[comment]: # Potential analyses include:
[comment]: # 
[comment]: # - Price and return distributions
[comment]: # - Log returns
[comment]: # - Rolling volatility
[comment]: # - Autocorrelation
[comment]: # - Drawdowns
[comment]: # - Correlation across assets
[comment]: # - Tail behavior
[comment]: # - Distributional assumptions
[comment]: # - Volatility clustering
[comment]: # - Market-regime comparisons
[comment]: # 
[comment]: # The goal is to identify which statistical properties of real financial data should be reproduced by later simulation and modeling steps.
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## 2. Stock-Price Path Simulation
[comment]: # 
[comment]: # The repository will compare observed stock behavior with simulated paths generated from common stochastic models.
[comment]: # 
[comment]: # Initial models may include:
[comment]: # 
[comment]: # ### Geometric Brownian Motion
[comment]: # 
[comment]: # $$ dS_t = \mu S_t dt + \sigma S_t dW_t $$
[comment]: # 
[comment]: # GBM provides a useful baseline for studying the consequences of assuming constant drift, constant volatility, and normally distributed log returns.
[comment]: # 
[comment]: # ### Historical Bootstrap
[comment]: # 
[comment]: # Historical returns can also be resampled to generate paths without imposing a normal return distribution.
[comment]: # 
[comment]: # ### Time-Varying Volatility Models
[comment]: # 
[comment]: # Extensions may include models such as:
[comment]: # 
[comment]: # - GARCH
[comment]: # - Regime-dependent volatility
[comment]: # - Stochastic volatility
[comment]: # 
[comment]: # Simulated paths can then be compared using quantities such as:
[comment]: # 
[comment]: # - Return distributions
[comment]: # - Realized volatility
[comment]: # - Drawdowns
[comment]: # - Extreme moves
[comment]: # - Path-dependent statistics
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## 3. Option Pricing
[comment]: # 
[comment]: # The project connects stock-price dynamics to derivative valuation.
[comment]: # 
[comment]: # Initial methods may include:
[comment]: # 
[comment]: # ### Black-Scholes
[comment]: # 
[comment]: # European option prices can be computed using the Black-Scholes model and used as a theoretical benchmark.
[comment]: # 
[comment]: # ### Monte Carlo Pricing
[comment]: # 
[comment]: # Simulated stock paths can be used to estimate option values numerically.
[comment]: # 
[comment]: # This allows direct comparison between:
[comment]: # 
[comment]: # - Analytical prices
[comment]: # - Monte Carlo estimates
[comment]: # - Market prices
[comment]: # - Machine-learning predictions
[comment]: # 
[comment]: # Additional analyses may examine:
[comment]: # 
[comment]: # - Calls and puts
[comment]: # - Moneyness
[comment]: # - Time to expiration
[comment]: # - Implied volatility
[comment]: # - Greeks
[comment]: # - Volatility smiles and skews
[comment]: # - Sensitivity to model assumptions
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## 4. Machine Learning
[comment]: # 
[comment]: # Machine-learning models will be used to study financial prediction problems such as:
[comment]: # 
[comment]: # - Future returns
[comment]: # - Future volatility
[comment]: # - Option prices
[comment]: # - Implied volatility
[comment]: # - Directional price movement
[comment]: # 
[comment]: # Candidate models include:
[comment]: # 
[comment]: # - Linear regression
[comment]: # - Logistic regression
[comment]: # - Regularized regression
[comment]: # - Random forests
[comment]: # - Gradient-boosted trees
[comment]: # - Neural networks
[comment]: # 
[comment]: # The focus is not simply on using increasingly complex models. Every model will be compared against simple financial and statistical baselines.
[comment]: # 
[comment]: # ### Example Features
[comment]: # 
[comment]: # Potential predictors include:
[comment]: # 
[comment]: # - Lagged returns
[comment]: # - Rolling volatility
[comment]: # - Moving averages
[comment]: # - Momentum indicators
[comment]: # - Volume
[comment]: # - Drawdown statistics
[comment]: # - Market index returns
[comment]: # - Option moneyness
[comment]: # - Time to expiration
[comment]: # - Risk-free rates
[comment]: # - Historical volatility
[comment]: # - Implied volatility
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## 5. Statistical Analysis
[comment]: # 
[comment]: # Model performance will be evaluated using more than a single predictive metric.
[comment]: # 
[comment]: # Depending on the problem, analyses may include:
[comment]: # 
[comment]: # - Confidence intervals
[comment]: # - Bootstrap uncertainty
[comment]: # - Hypothesis tests
[comment]: # - Residual diagnostics
[comment]: # - Out-of-sample error
[comment]: # - Time-series cross-validation
[comment]: # - Distribution comparisons
[comment]: # - Correlation analysis
[comment]: # - Calibration
[comment]: # - Statistical significance of model improvements
[comment]: # 
[comment]: # For example, if two models have different test errors, the project will investigate whether that difference is large relative to the statistical uncertainty of the evaluation.
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## 6. Statistical Interpretation
[comment]: # 
[comment]: # A central goal of the repository is to distinguish between:
[comment]: # 
[comment]: # > **A model producing a better metric**
[comment]: # 
[comment]: # and
[comment]: # 
[comment]: # > **Evidence that the model has learned a stable and economically meaningful relationship**
[comment]: # 
[comment]: # Model interpretation may include:
[comment]: # 
[comment]: # - Feature importance
[comment]: # - Permutation importance
[comment]: # - Partial dependence
[comment]: # - SHAP values
[comment]: # - Residual analysis
[comment]: # - Prediction distributions
[comment]: # - Error decomposition
[comment]: # - Performance across volatility regimes
[comment]: # - Performance across market periods
[comment]: # 
[comment]: # This allows the project to investigate questions such as:
[comment]: # 
[comment]: # - Does a model depend heavily on recent volatility?
[comment]: # - Does performance disappear during high-volatility periods?
[comment]: # - Are improvements concentrated in only a small number of observations?
[comment]: # - Do predictions remain stable when the data distribution changes?
[comment]: # - Are statistically significant improvements economically meaningful?
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## 7. Model Validation
[comment]: # 
[comment]: # Financial machine learning is especially vulnerable to overfitting and data leakage.
[comment]: # 
[comment]: # The project therefore emphasizes validation practices such as:
[comment]: # 
[comment]: # - Chronological train-validation-test splits
[comment]: # - Walk-forward validation
[comment]: # - Time-series cross-validation
[comment]: # - Feature construction using only past information
[comment]: # - Out-of-sample evaluation
[comment]: # - Comparison against naive baselines
[comment]: # 
[comment]: # Random train-test splitting will generally be avoided when it would allow future market information to influence past predictions.
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## Example Research Workflow
[comment]: # 
[comment]: # A typical experiment in this repository may follow the sequence:
[comment]: # 
[comment]: # ```text
[comment]: # Historical Market Data
[comment]: #         ↓
[comment]: # Exploratory Statistical Analysis
[comment]: #         ↓
[comment]: # Feature Engineering
[comment]: #         ↓
[comment]: # Baseline Financial Model
[comment]: #         ↓
[comment]: # Machine-Learning Model
[comment]: #         ↓
[comment]: # Out-of-Sample Evaluation
[comment]: #         ↓
[comment]: # Statistical Comparison
[comment]: #         ↓
[comment]: # Model Interpretation
[comment]: #         ↓
[comment]: # Economic Interpretation
[comment]: # ```
[comment]: # 
[comment]: # This structure keeps prediction, statistical evidence, and financial interpretation connected.
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## Example Research Questions
[comment]: # 
[comment]: # Some experiments planned for the repository include:
[comment]: # 
[comment]: # - How different are empirical stock-return distributions from the assumptions of GBM?
[comment]: # - Does a GARCH volatility model generate more realistic stock paths than constant-volatility GBM?
[comment]: # - How much Monte Carlo uncertainty is present in simulated option prices?
[comment]: # - Can machine learning reproduce Black-Scholes prices from option inputs?
[comment]: # - Can machine learning improve option-price predictions when market prices deviate from simplified theoretical assumptions?
[comment]: # - Which variables contribute most strongly to predicted implied volatility?
[comment]: # - Does additional model complexity produce statistically significant out-of-sample improvements?
[comment]: # - How does model performance change between low- and high-volatility market regimes?
[comment]: # - How stable are feature importances over time?
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## Evaluation Metrics
[comment]: # 
[comment]: # Depending on the problem, model performance may be evaluated with:
[comment]: # 
[comment]: # ### Regression
[comment]: # 
[comment]: # - Mean Absolute Error
[comment]: # - Mean Squared Error
[comment]: # - Root Mean Squared Error
[comment]: # - $R^2$
[comment]: # - Relative pricing error
[comment]: # 
[comment]: # ### Classification
[comment]: # 
[comment]: # - Accuracy
[comment]: # - Precision
[comment]: # - Recall
[comment]: # - F1 score
[comment]: # - ROC-AUC
[comment]: # 
[comment]: # ### Financial Evaluation
[comment]: # 
[comment]: # Where appropriate, statistical metrics may be supplemented with quantities such as:
[comment]: # 
[comment]: # - Directional accuracy
[comment]: # - Volatility forecast error
[comment]: # - Option-pricing error
[comment]: # - Return distributions
[comment]: # - Drawdown behavior
[comment]: # - Risk-adjusted performance
[comment]: # 
[comment]: # Predictive performance alone will not automatically be interpreted as evidence of a profitable trading strategy.
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## Technologies
[comment]: # 
[comment]: # The project is primarily implemented in Python.
[comment]: # 
[comment]: # Core libraries may include:
[comment]: # 
[comment]: # ```text
[comment]: # NumPy
[comment]: # Pandas
[comment]: # SciPy
[comment]: # Matplotlib
[comment]: # scikit-learn
[comment]: # statsmodels
[comment]: # PyTorch
[comment]: # yfinance
[comment]: # tensorflow
[comment]: # keras
[comment]: # ```
[comment]: # 
[comment]: # Additional packages may be introduced as the project develops.
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## Installation
[comment]: # 
[comment]: # Clone the repository:
[comment]: # 
[comment]: # ```bash
[comment]: # git clone https://github.com/luisfelipegz/quantflux.git
[comment]: # cd quantflux
[comment]: # ```
[comment]: # 
[comment]: # Create a virtual environment:
[comment]: # 
[comment]: # ```bash
[comment]: # python -m venv .venv
[comment]: # ```
[comment]: # 
[comment]: # Activate it on macOS or Linux:
[comment]: # 
[comment]: # ```bash
[comment]: # source .venv/bin/activate
[comment]: # ```
[comment]: # 
[comment]: # Activate it on Windows:
[comment]: # 
[comment]: # ```bash
[comment]: # .venv\Scripts\activate
[comment]: # ```
[comment]: # 
[comment]: # Install dependencies:
[comment]: # 
[comment]: # ```bash
[comment]: # pip install -r requirements.txt
[comment]: # ```
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## Current Development Plan
[comment]: # 
[comment]: # ### Phase 1
[comment]: # - Historical stock-data pipeline
[comment]: # - Exploratory return and volatility analysis
[comment]: # - GBM stock-path simulation
[comment]: # - Monte Carlo visualization
[comment]: # 
[comment]: # ### Phase 2
[comment]: # - Black-Scholes implementation
[comment]: # - Monte Carlo option pricing
[comment]: # - Greeks and implied-volatility analysis
[comment]: # - Comparison of simulated and theoretical prices
[comment]: # 
[comment]: # ### Phase 3
[comment]: # - Machine-learning baselines
[comment]: # - Feature engineering
[comment]: # - Time-series validation
[comment]: # - Model comparison
[comment]: # 
[comment]: # ### Phase 4
[comment]: # - Statistical uncertainty analysis
[comment]: # - Bootstrap comparisons
[comment]: # - Feature interpretation
[comment]: # - Regime-dependent performance analysis
[comment]: # 
[comment]: # ### Phase 5
[comment]: # - More realistic volatility models
[comment]: # - Extended option datasets
[comment]: # - Robustness studies
[comment]: # - End-to-end research case studies
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## Reproducibility
[comment]: # 
[comment]: # Where possible, experiments will include:
[comment]: # 
[comment]: # - Fixed random seeds
[comment]: # - Explicit model parameters
[comment]: # - Saved configuration settings
[comment]: # - Reusable data-processing functions
[comment]: # - Separate training and evaluation code
[comment]: # - Reproducible figures and tables
[comment]: # 
[comment]: # The objective is for results to be easy to verify and extend.
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## Disclaimer
[comment]: # 
[comment]: # This repository is intended for **research and educational purposes only**.
[comment]: # 
[comment]: # Nothing in this project should be interpreted as financial advice, an investment recommendation, or evidence that a particular strategy will remain profitable in future markets.
[comment]: # 
[comment]: # ---
[comment]: # 
[comment]: # ## Author
[comment]: # 
[comment]: # **Luis Felipe Gutierrez**
[comment]: # 
[comment]: # Physics PhD transitioning quantitative research, statistical modeling, and machine learning methods toward financial applications.
[comment]: # 
[comment]: # GitHub: [github.com/luisfelipegz](https://github.com/luisfelipegz)
