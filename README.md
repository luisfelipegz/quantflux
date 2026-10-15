# Quantflux

Quantflux is a Python-based quantitative finance research project focused on financial data analysis, statistical modeling, simulations, and derivatives pricing.

The project is being developed incrementally, with an emphasis on reusable code, numerical accuracy, and reproducible analysis.

## Current Functionality

### Historical Market Data

Historical stock market data is retrieved using `yfinance`, supporting multiple financial instruments.

The data is organized using Pandas DataFrames with financial variables on the first column level and ticker symbols on the second.

### Financial Calculations

The following variables are currently supported:

- Daily simple returns
- Daily gross returns
- Daily logarithmic returns
- Cumulative investment growth
- Cumulative returns
- Rolling annualized volatility
- Running price peaks
- Historical drawdowns

These calculations currently use adjusted closing prices.

### Data Validation

Historical datasets are checked for missing or invalid adjusted closing prices, duplicate dates, and structural inconsistencies before financial analysis.

### Testing

Automated unit tests verify financial calculations using predefined mathematical examples and check the handling of invalid market data.

### Market Visualization

The first Jupyter notebook includes visualizations of:

- Historical adjusted closing prices
- Cumulative returns
- Rolling annualized volatility
- Historical drawdowns

The notebook also includes numerical verification and financial consistency checks.

## Project Structure

```text
quantflux/
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── yfinance_provider.py
│       └── validation.py
├── tests/
│   ├── __init__.py
│   ├── test_market_features.py
│   └── test_validation.py
├── notebooks/
│   └── 01_market_exploration.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

1. Create a Python environment and install the packages listed in `requirements.txt`.
2. Open the project in Visual Studio Code.
3. Open the `notebooks` directory in JupyterLab.
4. Select the appropriate Python environment.
5. Open `01_market_exploration.ipynb`.
6. Execute the notebook cells to retrieve market data, calculate financial variables, and generate visualizations.

Automated tests can be executed using the Testing panel in Visual Studio Code.

## Future Development

Planned extensions include:

- Statistical analysis and financial distributions
- Stochastic stock-price simulations
- Multiperiod binomial models
- Black–Scholes option pricing
- Numerical methods for derivatives pricing
- Machine learning for financial analysis
- Model evaluation and statistical interpretation

## Data Source

Historical market data is provided through Yahoo Finance using the `yfinance` Python library.
