from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Time series forecasting is an important research topic in machine learning due to its prevalence in social and scientific applications. Multi-model forecasting paradigm, including model hybridization and model combination, is shown to be more effective than single-model forecasting in the M4 competition. In this study, we hybridize exponential smoothing with transformer architecture to capture both levels and seasonal patterns while exploiting the complex non-linear trend in time series data. We show that our model can capture complex trends and seasonal patterns with moderately improvement in comparison to the state-of-the-arts result from the M4 competition.',
    author='Sang TTruong',
    license='MIT',
)
