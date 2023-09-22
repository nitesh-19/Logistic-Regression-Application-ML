import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
import math
import time


def apply_logistic_regression(W, X, B):
    """
    Runs logistic regression on given parameters.

    :param W: Weights as a numpy array.
    :param X: Features as a numpy array.
    :param B: The offset term supplied as a float or an integer.
    :return: The output of the Logistic Regressio Function between 1 and 0.
    """
    frame = W * X
    if len(frame.shape) == 1:
        addition_column = np.sum(frame)
    else:
        addition_column = np.sum(frame, axis=1)
    return 1 / (1 + np.exp(-(addition_column + B)))


def binomial_equation(W, X, B):
    return -((W[0] * X + B) / W[1])


class LogisticRegression:
    def __init__(self, data_path, feature_indexes, target_index, alpha, iterations_limit=10000, values_to_replace=None,
                 create_test_set=False):
        """

        :param data_path:
        :param feature_indexes:
        :param target_index:
        :param values_to_replace:
        """
        self.range_of_rows = None
        self.degree = None
        self.iterations_finished = None
        self.iterations_limit = iterations_limit
        self.test_set = None
        self.should_make_test_set = create_test_set
        self.cost = None
        self.alpha = alpha
        self.scaling_factors = []
        self.feature_names = None
        self.working_dataframe = None
        self.DATA_PATH = data_path
        self.feature_column_indexes = feature_indexes
        self.target_index = target_index
        self.values_to_replace = values_to_replace
        self.m = None
        #########
        self.W = np.zeros(len(self.feature_column_indexes))
        # self.W = np.array([7.06308726, 5.40859122])
        self.B = 0
        # self.B = -6.790020028258589
        #######
        self.target_name = None
        self.build_training_dataframe()
        self.run_trainer()
        # print(self.predict_from_saved_model(feature_values=[7, 120]))
        # self.plot_curve()

    def plot_curve(self):
        x_coordinates = []
        y_coordinates = []
        z_coordinates = []
        for i in range(90):
            i = i / 10
            for j in range(50):
                j = j * 2
                x_coordinates.append(i)
                y_coordinates.append(j)
                z_coordinate = self.predict_from_saved_model([i, j])  # * self.scale_factors[1])
                if z_coordinate >= 0.5:
                    z_coordinates.append(2)
                else:
                    z_coordinates.append(0)
        fig1 = plt.figure("figure 1")
        plt.title("data")
        plt.scatter(x_coordinates, y_coordinates, s=z_coordinates)
        # plt.scatter(dataframe["hours"], dataframe["iq"], c=dataframe["result"], marker="x")
        plt.show()

    def plot_2d_result(self, columns):
        self.scale_data(unscale=True)
        column_1 = np.linspace(self.working_dataframe[columns[0]].min(), self.working_dataframe[columns[0]].max(), 15)
        column_2 = np.linspace(self.working_dataframe[columns[1]].min(), self.working_dataframe[columns[1]].max(), 15)
        z_coordinates = np.zeros((len(column_2), len(column_1)))
        for j in range(len(column_2)):
            temp_z = np.zeros(len(column_1))
            for i in range(len(column_1)):
                z = self.predict_from_saved_model([column_1[i], column_2[j]])
                # if z >= 0.5:
                #     temp_z[i] = 1
                # elif z < 0.5:
                #     temp_z[i] = 0
                temp_z[i] = z
            z_coordinates[j] = temp_z

        fig1 = plt.figure("figure 1")
        plt.title("Decision Boundary")
        plt.contourf(column_1, column_2, z_coordinates)
        plt.scatter(self.working_dataframe[columns[0]], self.working_dataframe[columns[1]],
                    c=self.working_dataframe[columns[2]], marker="x")

        # fig2 = plt.figure("figure 2")
        # plt.title("Cost")
        # plt.plot(X, Y, "o")
        plt.show()

    def predict_from_saved_model(self, feature_values, model_number=-1):
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
                feature_values = np.array(feature_values)
                feature_values = np.divide(feature_values, scale_factors)
                result = apply_logistic_regression(W, self.map_features(feature_values), B)
                return result

    def create_test_set(self, data, percent_of_data=20):
        """
        Create a testing dataset from the dataframe supplied at random.
        :param data: Pandas Dataframe
        :param percent_of_data: Integer. Percentage of data to keep aside for testing.
        :return: Pandas Dataframe.
        """

        rows_of_data = data.shape[0]
        random_indexes = sample(range(rows_of_data), round(rows_of_data * percent_of_data / 100))
        test_set = pd.DataFrame(data=data, index=random_indexes)
        data.drop(index=random_indexes, inplace=True)
        test_set.to_csv("test_set.csv")
        data.to_csv("training_set.csv")

        self.test_set = test_set
        return data

    def map_dataframe(self, degree=6):
        """
        Increases the training dataframe size by extrapolating higher degree features from already existing features to
        better fit the data.

        :param degree: Integer. The degree of the polynomial function that will fit the data.
        :return: None
        """
        self.degree = degree
        # Temporary dataframe to organize data.
        mapped_dataframe = pd.DataFrame(columns=self.feature_names)

        # Create empty columns for original and mapped features.
        mapped_dataframe[
            [i for i in range(len(self.feature_names),
                              len(self.feature_names) * degree + math.comb(degree, 2))]] = np.nan

        # Create and add mapped features
        for row_index in range(self.m):
            added_features = []
            features = self.get_features_from_row(row_index)
            ##            #####
            for i in range(1, degree + 1):
                for j in range(i + 1):
                    added_features.append((features[0] **  (i - j) * (features[1] ** j)))
            ##            #####
            mapped_dataframe.loc[row_index] = added_features

        mapped_dataframe[self.target_name] = self.working_dataframe[self.target_name]

        # Replace the original dataframe with mapped dataframe
        self.working_dataframe = mapped_dataframe
        self.feature_names = self.working_dataframe.columns[:-1]

        # Increase the number of weights to accommodate additional features
        self.W = np.zeros(mapped_dataframe.shape[1] - 1)

    def map_features(self, features_list):
        features = []
        for i in range(1, self.degree + 1):
            for j in range(i + 1):
                features.append((features_list[0] ** (i - j) * (features_list[1] ** j)))
        return np.array(features)

    def scale_data(self, unscale=False):
        """
        Scales the current dataframe by dividing each column with the maximum value in that column.

        :param unscale: Boolean. If true, the function scales the data back to original.
        :return: None
        """
        if not unscale:
            for i in range(len(self.feature_names)):
                max_value = self.working_dataframe[self.feature_names[i]].max()
                self.scaling_factors.append(max_value)

                self.working_dataframe[self.feature_names[i]] = \
                    self.working_dataframe[self.feature_names[i]] / max_value
        ############
        elif unscale:
            for i in range(2):
                self.working_dataframe[self.feature_names[i]] = \
                    self.working_dataframe[self.feature_names[i]] * self.scaling_factors[i]

    #############

    def build_training_dataframe(self):
        """
        Builds a training dataframe to run the algorithm on.
        Fetches data from csv file, adds higher degree features, deletes rows with missing data and also scales the data.
        :return: None
        """

        if self.DATA_PATH is None:
            raise TypeError(
                f"LinearRegression.DATA_PATH empty. Try setting LinearRegression.DATA_PATH to a string before starting "
                f"the training.")
        if self.feature_column_indexes is None:
            raise TypeError(
                f"LinearRegression.x_index_list empty. Try setting LinearRegression.x_index_list to a list before "
                f"starting the training.")
        if self.target_index is None:
            raise TypeError(
                f"LinearRegression.y empty. Try setting LinearRegression.y to a list before starting the "
                f"training.")
        else:
            data = pd.read_csv(self.DATA_PATH)

            # Replace the alphabetical data with the numerical data.
            if self.values_to_replace is not None:
                data = self.replace_strings_with_numbers(data=data)

            # Create Dataframe
            self.feature_names = [data.columns[index] for index in self.feature_column_indexes]
            self.target_name = data.columns[self.target_index]
            self.working_dataframe = pd.DataFrame(data=data[self.feature_names]).copy()
            self.working_dataframe[self.target_name] = data[self.target_name]
            self.working_dataframe.dropna(axis=0, inplace=True)

            # Separate testing data from the dataframe.
            if self.should_make_test_set is True:
                self.working_dataframe = self.create_test_set(self.working_dataframe, percent_of_data=20)
            self.m = self.working_dataframe.shape[0]
            self.scale_data()
            self.map_dataframe(degree=6)

    def replace_strings_with_numbers(self, data):
        """
        Replaces the string values in dataframe with corresponding numerical values as supplied in the
        "values_to_replace" argument.

        :param data: Pandas Dataframe to replace the values in.
        :return: The Dataframe after replacing the values.
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
            X_array = self.map_features(
                self.get_features_from_row(index=i, dataframe=test_data) / self.scaling_factors)
            y = apply_logistic_regression(self.W, X_array, self.B)
            if y >= 0.5:
                y = 1
            else:
                y = 0
            if test_data[self.target_name][i] == y:
                number_of_correct_predictions += 1
        return round(number_of_correct_predictions / test_data.shape[0] * 100, 2)

    def cost_function(self):
        """
        Calculates and returns overall cost of the model with the current parameters.

        :return: (Float) Cost of the model
        """
        # start = time.time()
        X_array = self.working_dataframe[self.feature_names].to_numpy()
        y_i = self.working_dataframe[self.target_name]

        sum_of_dataframe = (y_i * np.log(apply_logistic_regression(self.W, X_array, self.B))) + (
                (1 - y_i) * np.log(1 - apply_logistic_regression(self.W, X_array, self.B)))
        sum_of_dataframe = np.sum(sum_of_dataframe)

        self.cost = sum_of_dataframe * (-1 / self.m)
        # end = time.time()
        # print("The time of execution for cost_function :",
        #       (end - start) * 10 ** 3, "ms")
        return self.cost

    def gradient_descent(self):
        """
        Runs Gradient descent on the model and updates the weights.
        :return: None
        """

        # start = time.time()
        sum_of_W = np.zeros((self.range_of_rows[-1], len(self.feature_names)))
        X_array = self.working_dataframe[self.feature_names].to_numpy()
        w_frame = np.tile(self.W, (self.range_of_rows[-1], 1))
        term1 = np.array(apply_logistic_regression(w_frame, X_array, self.B) -
                         self.working_dataframe[self.target_name])
        sum_of_W += term1[:, np.newaxis] * X_array
        sum_of_W = np.sum(sum_of_W, axis=0)
        sum_of_W /= self.m
        sum_of_W *= self.alpha

        sum_of_B = np.sum(term1)

        # Update Parameters
        self.W = self.W - sum_of_W
        sum_of_B /= self.m
        reduction_term_B = sum_of_B * self.alpha
        self.B = self.B - reduction_term_B
        # end = time.time()
        # print("The time of execution for gradient_descent :",
        #       (end - start) * 10 ** 3, "ms")
        self.cost_function()

    def get_features_from_row(self, index, dataframe=None):
        """
        Returns a Numpy array of all the features in a specific row of the dataframe.
        :param dataframe:
        :param index: The index of the row to be extracted
        :return: Numpy array of the all the features in the row
        """
        if dataframe is None:
            return np.array(self.working_dataframe.iloc[index][self.feature_names])
        else:
            width = dataframe.shape[1] - 1
            return np.array(dataframe.iloc[index][[i for i in range(width)]])

    def write_json(self):
        with open("models.json", "r") as inpfile:
            try:
                data = json.load(inpfile)
                list_data = [int(key) for key in list(data)]
                params = {max(list_data) + 1: {"W": list(self.W),
                                               "B": str(self.B),
                                               "scale_factors": str(self.scaling_factors),
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
        self.range_of_rows = range(self.m + 1)
        should_continue = True
        prev_iteration_cost = 0
        self.iterations_finished = 0
        cost_history = []
        iterations_history = []
        try:
            while should_continue and self.iterations_finished <= self.iterations_limit:
                self.gradient_descent()
                if self.iterations_finished % 50 == 0:
                    print(f"W: {self.W}, B: {self.B}, Cost: {self.cost}")
                self.iterations_finished += 1

                # Sampling to plot cost function graph
                if self.iterations_finished % 100 == 0:
                    cost_history.append(self.cost)
                    iterations_history.append(self.iterations_finished)

                if self.cost == prev_iteration_cost:
                    print(self.W, self.B, self.cost)
                    should_continue = False
                prev_iteration_cost = self.cost

        except KeyboardInterrupt:
            self.write_json()
            self.plot_2d_result(columns=["hours", "iq", "result"])
            print(f"Accuracy of the model: {self.get_model_accuracy()}%")
        else:
            self.write_json()
            self.plot_2d_result(columns=["hours", "iq", "result"])
            print(f"Accuracy of the model: {self.get_model_accuracy()}%")
