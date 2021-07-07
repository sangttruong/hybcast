# Time-series Forecasting

Time series forecasting is an important research topic in machine learning due to its prevalence in social and scientific applications. Multi-model forecasting paradigm, including model hybridization and model combination, is shown to be more effective than single-model forecasting in the M4 competition. In this study, we hybridize exponential smoothing with transformer architecture to capture both levels and seasonal patterns while exploiting the complex non-linear trend in time series data. We show that our model can capture complex trends and seasonal patterns with moderately improvement in comparison to the state-of-the-arts result from the M4 competition.

<!-- Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p> -->

## Requirement
* Python >= 3.5
* PyTorch >= 1.7
* Numpy >= 1.19
* Matplotlib >= 3.3

## Usage
The base code of [ESRNN](https://github.com/kdgutier/esrnn_torch) and [Transformer](https://github.com/maxjcohen/transformer) is in Python. Therefore, for convienient purpose in doing Time Series forcasting research, we also implement ESTransformer in Python. The user can download the data from [M4 Competition data](https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/), using our code. Below is a list of currently supported components:
* Naive:  This benchmark model produces a forecast that is equal to the last observed value for a given time series.
* Seasonal Naive:  This benchmark model produces a forecast that is equal to
  the last observed value of the same season for a given time series.
* Naive2: a popular benchmark model for time series forecasting that automatically adapts
  to the potential seasonality of a series based on an autocorrelation test.
  If the series is seasonal, the model composes the predictions of Naive and Seasonal Naive. Otherwise, the model predicts on the simple Naive.
* Exponential smoothing: Using multiplicative Holt-Winter exponential smoothing to capture the potential error, seasonal, and trend.  
* Transformer: Using time-series transformer to optimize the trend.

<div style="text-align:center">
<img src= "./reports/figures/estransformer - main.png" width=50%/>
</div>

To reproduce an experiment, run the following command: 

```console
cd hybcast/run
bash experiment1.sh # run a single experiment without gpu
```

As the project is continue to evolve, please direct any question, feedback, or comment to [sttruong@stanford.edu](sttruong@stanford.edu).

## Acknowledgement
This research has been conducted as a part of internship of Sang Truong at Cummins Inc. (Fall 2019, Winter 2020), Community Health Network Inc. (Fall 2020, Spring 2021), and as indepedent research at DePauw University under the mentorship of Professor Jeff Gropp (Spring 2018, Fall 2019, Spring 2019, Fall 2020, Spring 2021). We thank Shuto Araki for his collaboration during Spring 2018 on theory and implementation of ARIMA and KNN models. We thank Bu Tran for his work on testing the ESTransformer architectures during Spring 2021.

## Citation
```
@inproceedings{
    truong2021hybcast,
    title={Time-series Forecasting},
    author={Sang Truong and Jeffrey Gropp},
    booktitle={},
    year={2021},
    url={}
}
```