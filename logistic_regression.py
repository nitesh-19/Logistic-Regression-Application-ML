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
        self.cost = None
        self.alpha = 0.01
        self.feature_column_names_list = None
        self.working_dataframe = None
        self.DATA_PATH = data_path
        self.features_index_list = feature_indexes
        self.target_index_list = target_index
        self.values_to_replace = values_to_replace
        self.m = None
        self.W = np.zeros(len(self.features_index_list))
        self.B = 0

        self.target_column_name = None
        self.build_training_dataframe()
        print(self.get_feature_array(0))
        self.run_trainer()

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
            if self.values_to_replace is not None:
                data = self.replace_strings_with_numbers(data=data)
            self.feature_column_names_list = [data.columns[index] for index in self.features_index_list]
            self.target_column_name = data.columns[self.target_index_list]
            self.working_dataframe = pd.DataFrame(data=data[self.feature_column_names_list]).copy()
            self.working_dataframe[self.target_column_name] = data[self.target_column_name]
            self.working_dataframe.dropna(axis=0, inplace=True)
            self.m = self.working_dataframe.shape[0]

    def replace_strings_with_numbers(self, data):
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

    def cost_function(self):
        summation = 0
        for integer in range(0, self.m):
            summation += self.working_dataframe.iloc[integer][self.target_column_name] * np.log10(
                apply_logistic_regression(self.W, self.get_feature_array(integer), self.B)) + (
                                 1 - self.working_dataframe.iloc[integer][self.target_column_name]) * np.log10(
                1 - apply_logistic_regression(self.W, self.get_feature_array(integer), self.B))
        self.cost = summation * (-1 / self.m)
        return self.cost

    def gradient_descent(self):
        # temp_W = self.W - self.alpha*()
        sum_of_W = 0
        sum_of_B = 0
        temp_W = np.zeros(len(self.features_index_list))

        for j in range(0, len(self.feature_column_names_list)):
            for i in range(0, self.m):
                X_array = self.get_feature_array(i)
                sum_of_W += apply_logistic_regression(self.W, X_array, self.B) - \
                            self.working_dataframe.iloc[i][self.target_column_name]* X_array[j]
            sum_of_W /= self.m
            reduction_term_W = sum_of_W * self.alpha
            temp_W[j] = self.W[j] - reduction_term_W

        for k in range(0, self.m):
            X_array = self.get_feature_array(k)
            sum_of_B += apply_logistic_regression(self.W, X_array, self.B) - \
                        self.working_dataframe.iloc[k][self.target_column_name]
            sum_of_B /= self.m
            reduction_term_B = sum_of_B * self.alpha
            temp_B = self.B - reduction_term_B

        self.W = temp_W
        self.B = temp_B
        self.cost_function()

    def get_feature_array(self, index):
        return np.array(self.working_dataframe.iloc[index][self.feature_column_names_list])

    def run_trainer(self):
        flag = True
        prev_cost = 0
        while flag:
            self.gradient_descent()
            print(self.W, self.B, self.cost)

            if self.cost == prev_cost:
                print(self.W, self.B, self.cost)
                flag = False
            prev_cost = self.cost
