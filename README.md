# ProjectTemplate

## Usage

Install the project directory using ```make install```

Use ```make --help``` to see other commands.

## Project Structure
```
├── LICENSE
├── .gitignore
├── Makefile                    <- Makefile with commands like `make data` or `make train`.
├── README.md                   <- The top-level README for developers using this project.
├── config
│   ├── model                   <- Specific configurations for the machine learning models, such as hyperparameters.
│   ├── process                 <- Configurations for data processing, including transformations and preprocessing steps.
│   └── main.yml                <- The main configuration file that aggregates the configurations from model and process.
│
├── data
│   ├── interim                 <- Intermediate data that has been transformed.
│   ├── processed               <- The final, canonical data sets for modeling.
│   └── raw                     <- The original, immutable data dump.
│
├── docs                        <- A default doc project.
│
├── models                      <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
│                               the creator's initials, and a short `-` delimited description, e.g.
│                               `1.0-jqp-initial-data-exploration`.
│
├── scripts                     <- Standalone scripts
│
├── references                  <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                     <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                 <- Generated graphics and figures to be used in reporting
│
├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
│                               generated with `pip freeze > requirements.txt`
│
├── setup.py                    <- Makes project pip installable (pip install -e .)
├── xrkit                         <- Source code for use in this project.
│   ├── __init__.py             <- Makes xrkit a Python module.
│   │
│   ├── data                    <- Functionality to download or generate data.
│   │   └── make_dataset.py
│   │
│   ├── models                  <- Functionality to train models and then use trained models to make
│   │   │                       predictions.
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
│   │                               generated with `pip freeze > requirements.txt`.
│   ├── requirements-test.txt   <- The requirements file with the necessary dependencies to run tests.
│   └── setup.py                <- Metadata about your project for easy distribution.
├── tests                       <- Test cases (named after module)
│   ├── data                    <- Data module tests.
│   │   └── make_dataset.py
│   └── models                  <- Models module tests.
│       ├── predict_model_test.py
│       └── train_model_test.py
└── setup.cfg                   <- Configuration file with settings for running the project tools.
````