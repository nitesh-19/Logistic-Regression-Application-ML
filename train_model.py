from logistic_regression import LogisticRegression

import pandas as pd

data_path = r".\data\train.csv"  # Dataset having features and target in a vertical arrangement.
feature_columns = [2, 4, 5, 9]  # Index of the columns in the dataset to be fed as features to the trainer
target_column = 1  # Index of the column in the dataset to be fed as the target to the trainer.

values_to_replace = {"Sex": {"male": 1, "female": 0},
                     }

test = LogisticRegression(data_path=data_path, feature_indexes=feature_columns, target_index=target_column,
                          values_to_replace=values_to_replace)
