# Python-project-1-Bosch-production-line

Consists of the solution for the competition Bosch Production Line Performance from kaggle (LINK TO DATASET: https://www.kaggle.com/c/bosch-production-line-performance/data)

The dataset is one of the largest datasets on kaggle in terms of features and hence it is processed and used in form of chunks

Matthews correlation coefficient(mcc) is used as the evaluation metric.

MCC obtained using the RandomForest is 0.41963 on private leaderboard.

### Data Description
The data for this competition represents measurements of parts as they move through Bosch's production lines. Each part has a unique Id. The goal is to predict which parts will fail quality control (represented by a 'Response' = 1).

The dataset contains an extremely large number of anonymized features. Features are named according to a convention that tells you the production line, the station on the line, and a feature number. E.g. L3_S36_F3939 is a feature measured on line 3, station 36, and is feature number 3939.

The dataset consists of three files numeric features, categorical features and date features. The list of important numeric features are extracted from the Numeric classifier notebook and is used in the main notebook along the date features.
