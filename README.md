# Predict-Student-Performance-From-Game-Play

This project focuses on predicting student performance in a game-based learning environment. The objective is to develop models that can accurately predict the correctness of students' answers during gameplay sessions. The project utilizes a dataset from the "[Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/overview)" competition on Kaggle.

## Dataset
The dataset used in this project contains information about game settings, user behavior, and assessment data from gameplay sessions. Each gameplay session consists of three sections, and at the end of each section, there is a checkpoint where a set of questions is given as an assessment. The dataset provides information up until the checkpoints, and user actions within the checkpoints are not recorded. The dataset is provided in CSV format and can be downloaded from the [Kaggle competition page](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/overview).

## Project Structure
The project is organized as follows:

- `data/`: This directory contains the dataset files. Note that train.csv and test.csv are not included and will be added once the competition dataset is public.
- `dataProcessing.R`: This file contains R scripts that prepare the data for analysis.
- `validation.R`: This file contains R scripts that perform model selection and validation.
- `report.Rmd`: This file contains R Markdown notebooks that showcase the data analysis, feature engineering, model training, and evaluation steps.
- `Train&Predict.R`:  This file contains R scripts that train the selected model on full training data, and predict outcomes on testing data.
- `README.md`: This file provides an overview of the project and instructions for running the code.

## Dependencies
The project has the following dependencies:
- R
- RStudio
- tidyverse
- caret
- randomForest
- glmnet
- xgboost

Make sure to install these dependencies before running the project. You can install the required packages using the install.packages() function in R.














