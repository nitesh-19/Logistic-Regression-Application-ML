import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def apply_logistic_regression(W, X, B):
    """
    Runs logistic regression on given parameters.

    :param W: Weights as a numpy array.
    :param X: Features as a numpy array.
    :param B: The offset term supplied as a float or an integer.
    :return: The output of the Logistic Regressio Function between 1 and 0.
    """
    return 1 / (1 + np.exp(-(np.dot(W, X)) + B))


class LogisticRegression:
    def __init__(self, data_path, feature_indexes, target_index, values_to_replace=None):
        """

        :param data_path:
        :param feature_indexes:
        :param target_index:
        :param values_to_replace:
        """
        self.working_dataframe = None
        self.DATA_PATH = data_path
        self.features_index_list = feature_indexes
        self.target_index_list = target_index
        self.values_to_replace = values_to_replace
        self.build_training_dataframe()

    def build_training_dataframe(self):
        """
        Builds a dataframe with only relevant features and target for the trainer to work on. Also drops all rows with
        missing data
        :return:
        """
        if self.DATA_PATH is None:
            raise TypeError(
                f"LinearRegression.DATA_PATH empty. Try setting LinearRegression.DATA_PATH to a string before starting "
                f"the training.")
        if self.features_index_list is None:
            raise TypeError(
                f"LinearRegression.x_index_list empty. Try setting LinearRegression.x_index_list to a list before "
                f"starting the training.")
        if self.target_index_list is None:
            raise TypeError(
                f"LinearRegression.y empty. Try setting LinearRegression.y to a list before starting the "
                f"training.")
        else:
            data = pd.read_csv(self.DATA_PATH)
            data.dropna(axis=0, inplace=True)
            if self.values_to_replace is not None:
                data = self.replace_values_with_numbers(data=data)
            feature_column_names_list = [data.columns[index] for index in self.features_index_list]
            target_column_name = data.columns[self.target_index_list]
            self.working_dataframe = pd.DataFrame(data=data[feature_column_names_list]).copy()
            self.working_dataframe[target_column_name] = data[target_column_name]

    def replace_values_with_numbers(self, data):
        """
        Replaces the string values in dataframe with corresponding numerical values as supplied in the
        "values_to_replace" argument.
        :param data: Dataframe to replace the values in
        :return: Dataframe after replacing the values
        """
        for key in self.values_to_replace:
            for inner_key in self.values_to_replace[key]:
                data[key].replace(inner_key, self.values_to_replace[key][inner_key], inplace=True)
        return data
