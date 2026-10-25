# Quantflux

Quantflux is a Python-based quantitative finance research project focused on financial data analysis, statistical modeling, simulations, and derivatives pricing.

The project is being developed incrementally, with an emphasis on reusable code, numerical accuracy, and reproducible analysis.

## Current Functionality

### Historical Market Data

Historical stock market data is retrieved using `yfinance`, supporting multiple financial instruments.

The data is organized using Pandas DataFrames with financial variables on the first column level and ticker symbols on the second.

Historical financial calculations currently use adjusted closing prices (`Adj Close`).

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

### Data Validation

Historical market datasets are validated before financial calculations. Current checks include:

- Empty datasets
- Invalid date indices
- Duplicate trading dates
- Missing adjusted closing prices
- Non-finite or non-positive adjusted closing prices
- Unexpected column structures

### Statistical Analysis

#### Descriptive Return Statistics

Quantflux supports descriptive analysis of historical
financial returns, including:

- Observation count
- Sample mean and standard deviation
- Skewness and excess kurtosis

#### Covariance and Correlation Analysis

Quantflux also supports analysis of relationships
between financial assets, including:

- Sample covariance matrices
- Pearson correlation matrices
- Simple and logarithmic returns

### Financial Visualization

Reusable Matplotlib functions support the following visualizations:

#### Market Performance

- Historical adjusted closing prices
- Cumulative returns
- Annualized rolling volatility
- Historical drawdowns

#### Return Distribution Analysis

- Return histograms with fitted Gaussian benchmarks
- Comparative box plots
- Normal Q-Q plots

#### Correlation Analysis

- Correlation heatmaps with numerical annotations
- Diverging color scale fixed between -1 and 1

### Testing and Numerical Verification

Quantflux uses Python's built-in unittest framework.

Automated tests cover:

- Financial return calculations
- Rolling volatility and drawdown calculations
- Market data validation
- Descriptive statistical calculations
- Financial visualization functions
- Independent mathematical verification of covariance calculations
- Independent mathematical verification of Pearson correlations
- Validation of correlation heatmaps, including matrix values, ticker labels, numerical annotations, and color scaling

Statistical calculations are verified against independent implementations of their mathematical definitions using synthetic datasets.

Visualization tests verify plotted values, ticker selection, figure structure, and relevant axis formatting.

### Interactive Financial Analysis

The Jupyter notebook `01_market_exploration.ipynb` demonstrates the use of Quantflux modules with historical market data.

The notebook includes:

- Historical data retrieval for AAPL, MSFT, and SPY
- Financial feature calculations
- Numerical consistency checks
- Market performance and volatility visualizations
- Descriptive return statistics
- Return distribution analysis using histograms, box plots, and Q-Q plots
- Covariance and correlation matrices
- Correlation heatmap with numerical annotations

## Project Structure

```text
quantflux/
├── src/
│   ├── data/
│   │   ├── yfinance_provider.py
│   │   └── validation.py
│   ├── analysis/
│   │   └── return_statistics.py
│   └── visualization/
│       └── market_plots.py
├── tests/
│   ├── test_market_features.py
│   ├── test_validation.py
│   ├── test_market_plots.py
│   └── test_return_statistics.py
├── notebooks/
│   └── 01_market_exploration.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### 1. Create a Python Environment

From the project root, create a virtual environment using Python 3.12:
```
python3 -m venv .venv
```

Activate the environment on macOS or Linux:
```
source .venv/bin/activate
```

### 2. Install Dependencies

Install the required Python packages:
```
python -m pip install -r requirements.txt
```

### 3. Open the Project

Open the Quantflux repository in Visual Studio Code and select the Python interpreter located inside `.venv`.

Open the `notebooks` directory in JupyterLab and ensure the notebook uses the same Python environment.

### 4. Run the Financial Analysis

Open:
```
notebooks/01_market_exploration.ipynb
```

Execute the notebook cells to retrieve historical data, calculate financial and statistical variables, and generate visualizations.

### 5. Run Automated Tests

From the project root, execute:
```
python -m unittest discover -s tests -v
```

The test suite verifies the implemented calculations, data validation, and visualization functionality.

## Future Development

Planned extensions include:

- Statistical hypothesis testing and distribution fitting
- Stochastic stock-price simulations
- Monte Carlo methods
- Multiperiod binomial option-pricing models
- Black–Scholes option pricing
- Numerical methods for derivatives pricing
- Financial risk modeling
- Machine learning applications
- Model evaluation and statistical interpretation

## Data Source

Historical market data is provided through Yahoo Finance using the `yfinance` Python library.

Data availability and historical values depend on the external data provider.
