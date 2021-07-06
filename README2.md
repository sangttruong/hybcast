# Time-series Forecasting

## Introduction
Time series forecasting is an important research topic in machine learning due to its prevalence in social and scientific applications. Multi-model forecasting paradigm, including model hybridization and model combination, is shown to be more effective than single-model forecasting in the M4 competition. In this study, we hybridize exponential smoothing with transformer architecture to capture both levels and seasonal patterns while exploiting the complex non-linear trend in time series data. We show that our model can capture complex trends and seasonal patterns with moderately improvement in comparison to the state-of-the-arts result from the M4 competition.

<!-- Multiple tactics were use to model complex seasionality in time-series, such as sinusodal model, seasional ARIMA, dynamic harmonic regression, BATS, and TBATS model. -->

## Requirements 
* Python >= 3.5
* Pytorch >= 1.9

## Usages

## Acknowledgement
This research has been conducted as a part of internship of Sang Truong at Cummins Inc. (Fall 2019, Winter 2020), Community Health Network Inc. (Fall 2020, Spring 2021), and as indepedent research at DePauw University under the mentorship of Professor Jeff Gropp (Spring 2018, Fall 2019, Spring 2019, Fall 2020, Spring 2021). We thank Shuto Araki for his collaboration during Spring 2018 on theory and implementation of ARIMA and KNN models. We thank Bu Tran for his work on testing the ESTransformer architectures during Spring 2021.
