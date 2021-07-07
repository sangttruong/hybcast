from setuptools import find_packages, setup

setup(
    name='hybcast',
    packages=find_packages(where = 'hybcast'),
    package_dir={'': 'hybcast'},
    # packages= find_packages() + find_packages(where="./hybcast/data") + find_packages(where="./hybcast/features")
                # + find_packages(where="./hybcast/models") + find_packages(where="./hybcast/visualization"),
    version='0.1.0',
    description='Time series forecasting is an important research topic in machine learning due to its prevalence in social and scientific applications. Multi-model forecasting paradigm, including model hybridization and model combination, is shown to be more effective than single-model forecasting in the M4 competition. In this study, we hybridize exponential smoothing with transformer architecture to capture both levels and seasonal patterns while exploiting the complex non-linear trend in time series data. We show that our model can capture complex trends and seasonal patterns with moderately improvement in comparison to the state-of-the-arts result from the M4 competition.',
    author='Sang TTruong',
    author_email="sttruong@stanford.edu",
    license='MIT',
    
    url="https://github.com/sangttruong/hybcast",
    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

    ],

    # packages=['hybcast',]
)
