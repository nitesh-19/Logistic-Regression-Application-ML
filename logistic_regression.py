import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample


def apply_logistic_regression(W, X, B):
    """
    Runs logistic regression on given parameters.

    :param W: Weights as a numpy array.
    :param X: Features as a numpy array.
    :param B: The offset term supplied as a float or an integer.
    :return: The output of the Logistic Regressio Function between 1 and 0.
    """

    return 1 / (1 + np.exp(-((np.dot(W, X)) + B)))


def binomial_equation(W, X, B):
    return -((W[0] * X + B) / W[1])


def predict_from_saved_model(features, model_number=-1):
    with open("models.json", "r") as input_file:
        try:
            data = json.load(input_file)
        except json.JSONDecodeError:
            print("No valid data found. Did you train the model first?")
        else:
            if model_number == -1:
                chosen_model = data[list(data)[-1]]
            else:
                chosen_model = data[str(model_number)]
            W = np.array(chosen_model["W"])
            B = eval(chosen_model["B"])
            scale_factors = eval(chosen_model["scale_factors"])
            features = np.array(features)
            features = np.divide(features, scale_factors)
            result = apply_logistic_regression(W, features, B)
            return result


class LogisticRegression:
    def __init__(self, data_path, feature_indexes, target_index, values_to_replace=None, create_test_set=False):
        """

        :param data_path:
        :param feature_indexes:
        :param target_index:
        :param values_to_replace:
        """
        self.iterations_finished = None
        self.test_set = None
        self.make_test_set = create_test_set
        self.cost = None
        self.alpha = 0.5
        self.scale_factors = []
        self.feature_column_names_list = None
        self.working_dataframe = None
        self.DATA_PATH = data_path
        self.features_index_list = feature_indexes
        self.target_index_list = target_index
        self.values_to_replace = values_to_replace
        self.m = None
        # self.W = np.zeros(len(self.features_index_list))
        self.W = np.array([7.06308726, 5.40859122])
        # self.B = 0
        self.B = -6.790020028258589
        self.target_column_name = None
        self.build_training_dataframe()
        self.run_trainer()

    def plot_result(self, dataframe, X, Y, W, B):
        x_coordinates = []
        y_coordinates = []
        for i in range(9):
            x_coordinates.append(i)
            y_coordinates.append(binomial_equation(W, i / self.scale_factors[0], B) * self.scale_factors[1])

        self.scale_data(unscale=True)

        fig1 = plt.figure("figure 1")
        plt.title("data")
        plt.plot(x_coordinates, y_coordinates, "o")
        plt.scatter(dataframe["hours"], dataframe["iq"], c=dataframe["result"], marker="x")

        fig2 = plt.figure("figure 2")
        plt.title("Cost")
        plt.plot(X, Y, "o")
        plt.show()

    def create_test_set(self, data, percent_of_data=20):
        length_of_data = len(data)
        list_of_random_index = sample(range(0, length_of_data), round(length_of_data * percent_of_data / 100))
        test_set = pd.DataFrame(data=data, index=list_of_random_index)
        data.drop(index=list_of_random_index, inplace=True)
        test_set.to_csv("test_set.csv")
        data.to_csv("training_set.csv")

        self.test_set = test_set
        return data

    def scale_data(self, unscale=False):
        if not unscale:
            for i in range(len(self.features_index_list)):
                max_value = self.working_dataframe[self.feature_column_names_list[i]].max()
                self.scale_factors.append(max_value)

                self.working_dataframe[self.feature_column_names_list[i]] = \
                    self.working_dataframe[self.feature_column_names_list[i]] / max_value

        elif unscale:
            for i in range(len(self.feature_column_names_list)):
                self.working_dataframe[self.feature_column_names_list[i]] = \
                    self.working_dataframe[self.feature_column_names_list[i]] * self.scale_factors[i]

            # self.B = self.B * self.scale_factors[-1]
            # for index in range(len(self.feature_column_names_list) - 1):
            #     self.W[index] = self.W[index] * self.scale_factors[-1] / self.scale_factors[index]

    def build_training_dataframe(self):
        """
        Builds a dataframe with only relevant features and target for the trainer to work on. Also drops all rows with
        missing data and scales the data
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
            if self.make_test_set is True:
                self.working_dataframe = self.create_test_set(self.working_dataframe)
            self.m = self.working_dataframe.shape[0]
            self.scale_data()

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

    def get_model_accuracy(self):
        test_data = pd.read_csv("test_set.csv")
        test_data.drop(test_data.columns[0], axis=1, inplace=True)
        number_of_correct_predictions = 0
        for i in range(test_data.shape[0]):
            X_array = self.get_feature_array(index=i, dataframe=test_data)
            X_array = X_array / self.scale_factors
            y = apply_logistic_regression(self.W, X_array, self.B)
            if y >= 0.5:
                y = 1
            else:
                y = 0
            if test_data[self.target_column_name][i] == y:
                number_of_correct_predictions += 1
        return round(number_of_correct_predictions / test_data.shape[0] * 100, 2)

    def cost_function(self):
        """
        Calculates and returns overall cost of the model with the current parameters.

        :return:
        """
        summation = 0
        for integer in range(self.m):
            y_i = self.working_dataframe.iloc[integer][self.target_column_name]
            X_array = self.get_feature_array(integer)
            if y_i == 0:
                summation += np.log(1 - apply_logistic_regression(self.W, X_array, self.B))
            elif y_i == 1:
                summation += np.log(apply_logistic_regression(self.W, X_array, self.B))
        self.cost = summation * (-1 / self.m)

        return self.cost

    def gradient_descent(self):
        """
        Runs Gradient descent on the model and updates the weights.
        :return:
        """
        sum_of_B = 0
        sum_of_W = np.zeros(len(self.features_index_list))
        for j in range(len(self.features_index_list)):
            for i in range(self.m):
                X_array = self.get_feature_array(i)
                sum_of_W[j] += (apply_logistic_regression(self.W, X_array, self.B) -
                                self.working_dataframe.iloc[i][self.target_column_name]) * X_array[j]
        sum_of_W /= self.m
        sum_of_W *= self.alpha

        for k in range(0, self.m):
            X_array = self.get_feature_array(k)

            sum_of_B += apply_logistic_regression(self.W, X_array, self.B) - self.working_dataframe.iloc[k][
                self.target_column_name]

        self.W = self.W - sum_of_W
        sum_of_B /= self.m
        reduction_term_B = sum_of_B * self.alpha
        self.B = self.B - reduction_term_B
        self.cost_function()

    def get_feature_array(self, index, dataframe=None):
        """
        Returns a Numpy array of all the features in a specific row of the dataframe.
        :param dataframe:
        :param index: The index of the row to be extracted
        :return: Numpy array of the all the features in the row
        """
        if dataframe is None:
            return np.array(self.working_dataframe.iloc[index][self.feature_column_names_list])
        else:
            return np.array(dataframe.iloc[index][self.feature_column_names_list])

    def write_json(self):
        with open("models.json", "r") as inpfile:
            try:
                data = json.load(inpfile)
                list_data = [int(key) for key in list(data)]
                params = {max(list_data) + 1: {"W": list(self.W),
                                               "B": str(self.B),
                                               "scale_factors": str(self.scale_factors),
                                               "iterations_finished": self.iterations_finished

                                               }}
                data.update(params)
            except json.JSONDecodeError:
                with open("models.json", "w") as outfile:
                    init_dictionary = {0: {"W": "init", "B": "init", }}
                    json.dump(init_dictionary, outfile, indent=4)
            else:
                with open("models.json", "w") as outfile:
                    json.dump(data, outfile, indent=4)

    def run_trainer(self):
        flag = True
        prev_cost = 0
        self.iterations_finished = 0
        cost_history = []
        iterations_history = []
        try:
            while flag:
                self.gradient_descent()
                print(f"W: {self.W}, B: {self.B}, Cost: {self.cost}")
                self.iterations_finished += 1
                if self.iterations_finished % 1 == 0:
                    cost_history.append(self.cost)
                    iterations_history.append(self.iterations_finished)

                if self.cost == prev_cost:
                    print(self.W, self.B, self.cost)
                    flag = False
                prev_cost = self.cost

        except KeyboardInterrupt:
            self.plot_result(self.working_dataframe, iterations_history, cost_history, self.W, self.B)
            print(f"Accuracy of the model: {self.get_model_accuracy()}%")
            self.write_json()
        else:
            self.plot_result(self.working_dataframe, iterations_history, cost_history, self.W, self.B)
            print(f"Accuracy of the model: {self.get_model_accuracy()}%")
            self.write_json()
