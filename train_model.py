from logistic_regression import LogisticRegression

import pandas as pd

data_path = r".\data\train.csv"  # Dataset having features and target in a vertical arrangement.
feature_columns = [0, 2, 3, 4]  # Index of the columns in the dataset to be fed as features to the trainer
target_column = 6  # Index of the column in the dataset to be fed as the target to the trainer.

