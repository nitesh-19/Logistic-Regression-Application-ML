from logistic_regression import LogisticRegression

data_path = r".\data\exam_data.csv"  # Dataset having features and target in a vertical arrangement.
feature_columns = [1, 2]  # Index of the columns in the dataset to be fed as features to the trainer
target_column = 3  # Index of the column in the dataset to be fed as the target to the trainer.

# values_to_replace = {"Sex": {"male": 0, "female": 1},
#                      }

# "18": {
#         "W": [
#             3.919056222214989,
#             3.8626644845090423,
#             0.8787077202813787,
#             10.124167455412513,
#             -0.5750696594156439,
#             -1.7402593150965935,
#             7.544067304883638,
#             5.786176476397604,
#             -2.943665154715387,
#             -2.7992691593608847,
#             4.949312960363013,
#             4.9389933736084615,
#             2.9071160586857454,
#             -3.414281557136126,
#             -2.8684675878608044,
#             3.3054702960155686,
#             3.5012359214964137,
#             2.870268288588516,
#             1.5414329139828717,
#             -3.0631897067931217,
#             -2.4773679924749357,
#             2.3803671585442947,
#             2.4569506872484976,
#             2.1013117628978555,
#             1.7571872745820734,
#             0.9679736185147684,
#             -2.4745571552763823
#         ],
#         "B": "-7.027972602359554",
#         "scale_factors": "[7.96, 100]",
#         "iterations_finished": 3214
#     }
test = LogisticRegression(data_path=data_path, feature_indexes=feature_columns, target_index=target_column,
                          values_to_replace=None, create_test_set=False)

# print(test.predict_from_saved_model(features=[4, 45]))
# print(test.map_features())
# test.map_features(degree=2)
