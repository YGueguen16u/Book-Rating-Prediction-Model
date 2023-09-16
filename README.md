# Project 1: Book Rating Prediction Model

## Introduction
This notebook aims to predict book ratings using various machine learning models. We will employ feature engineering techniques, data cleaning, and apply several regression algorithms to accomplish this task.

## Dependencies
To run this notebook, the following libraries need to be installed. You can install them using the requirements.txt file or by running the following code cell:

!pip install pandas numpy seaborn matplotlib scikit-learn

## The libraries used are:

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

## Notebook Structure
### Data Cleaning and Feature Engineering: In this section, we prepare the data for model training. This includes cleaning missing data, transforming features, and encoding categorical variables.

### Observation: This section focuses on the exploratory analysis of the data. We will use plots and tables to understand trends in the data.

### Machine Learning: This section is the core of the notebook where various regression models are applied and evaluated on the prepared dataset.

## How to Run the Project
To execute this notebook, start by installing the dependencies, and then run all the cells in sequence.

## Credits and License
This project was developed by Yann GUEGUEN. The project is under the MIT license, meaning you are free to copy, distribute, and modify it, provided you adhere to the terms of the MIT license.