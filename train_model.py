from logistic_regression import LogisticRegression, apply_logistic_regression

import pandas as pd

data_path = r".\data\exam_data.csv"  # Dataset having features and target in a vertical arrangement.
feature_columns = [1,2]  # Index of the columns in the dataset to be fed as features to the trainer
target_column = 3  # Index of the column in the dataset to be fed as the target to the trainer.

# values_to_replace = {"Sex": {"male": 0, "female": 1},
#                      }
# [ 0.17408336 -0.00558962], B: 0.0086609235924594, Cost: 0.28257570712380575
# W: [0.43008423 0.01532107], B: -2.3547165981004556, Cost: 0.43700016492113386
# W: [0.45786249 0.01740567], B: -2.5956953927710633, Cost: 0.4230503910212577
# W: [0.84721055 1.05935965], B: -6.713996364075926, Cost: 0.2736741464667875
test = LogisticRegression(data_path=data_path, feature_indexes=feature_columns, target_index=target_column,
                          values_to_replace=None)
# x = apply_logistic_regression([0.84721055, 1.05935965], [0, 0], -6.713996364075926)
# print(x)
# test.working_dataframe.to_csv("credte")
