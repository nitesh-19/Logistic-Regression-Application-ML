from logistic_regression import LogisticRegression

data_path = r".\data\exam_data.csv"  # Dataset having features and target in a vertical arrangement.
feature_columns = [1, 2]  # Index of the columns in the dataset to be fed as features to the trainer
target_column = 3  # Index of the column in the dataset to be fed as the target to the trainer.

# values_to_replace = {"Sex": {"male": 0, "female": 1},
#                      }

test = LogisticRegression(data_path=data_path, feature_indexes=feature_columns, target_index=target_column, alpha=0.5,
                           create_test_set=False, iterations_limit=70000)
test.plot_2d_result(columns=["iq", "hours", "result"])
# test.plot_2d_result(columns=["Age", "Fare", "Survived"])
# test.plot_2d_result(columns=["Fare", "Sex", "Survived"])

# print(test.predict_from_saved_model(features=[4, 45]))
# print(test.map_features())
# test.map_features(degree=2)
