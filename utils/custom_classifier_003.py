import pandas
import logging
import scipy
from scipy.special import softmax
from scipy.optimize import minimize, least_squares, root, basinhopping
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy
import sklearn.metrics  as metrics
import time
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import math
import pickle
import json

logging.basicConfig(filename="log.log")


#### Connection_Graph
#### Input_layer ####
# Input_hidden_feature_partition=list(int) of which sum = input layer size
# ie
#### Output_layer ####


class CustomMLPClassifier():
    def __init__(self, input_size, number_of_classes, number_of_hidden_features=None, half_window_size=10,
                 signal_batch_size=500000, hidden_feature_max_iter=10, classification_max_iter=25,
                 inverse_feature_max_iter=2, simple_test=False):
        #### Architecture:
        #### A. 1. Train: Feature Extractors (Timeshift Signal) -> Scalar Targets
        #### A. 2. Compute: Feature Extractors (Timeshift Signal) as Forward Features
        #### A. 3. Train: Final Classifer (Timeshift Signal + Forward Features) -> Categorical Target
        #### A. 4. Compute:  Final Classifer(Timeshift Signal + Forward Features) -> Categorical Output
        #### A. 5. Train: Inverse Features Generators(Timeshift Signal+ Categorical Output) -> Forward Features
        #### A. 6. Compute: Inverse Features Generators (Timeshift Signal+ Classifer Target ) -> Inverse Features

        #### B.1. Train: Feature Extractors (Timeshift Signal) -> Inverse Features
        #### B. 2. Compute: Feature Extractors (Timeshift Signal) as Better Forward Features
        #### B. 3. Train: Final Classifer (Timeshift Signal +More Realistic Forward Features) -> Categorical Target
        #### B. 4. Compute:  Final Classifer(Timeshift Signal + More Realistic Forward Features) -> Better Categorical Output
        #### B. 5. Train: Inverse Features Generators(Timeshift Signal+ Better Categorical Output ) -> Better Forward Features
        #### B. 6. Compute: Inverse Features Generators (Timeshift Signal+ Classifer Target ) -> Better Inverse Features

        #### Hidden feature = input_size * hidden_layers
        if number_of_hidden_features is None:
            number_of_hidden_features = 2 * number_of_classes

        self.number_of_hidden_features = number_of_hidden_features
        self.input_size = input_size
        self.number_of_classes = number_of_classes
        self.half_window_size = half_window_size
        self.hidden_feature_extractor_dict = dict()
        self.inverse_hidden_feature_generator_dict = dict()

        if simple_test:
            hidden_sizes_feature = (1,)
            hidden_sizes_final = (1 + number_of_hidden_features,)
            hidden_sizes_inverse_feature = (1,)
            hidden_feature_max_iter = 1
            classification_max_iter = number_of_hidden_features
            inverse_feature_max_iter = 1
        else:
            hidden_sizes_feature = (
                self.input_size + number_of_hidden_features, number_of_classes,)
            hidden_sizes_final = ((self.input_size + 2 * number_of_hidden_features),
                                  (self.input_size + 2 * number_of_hidden_features), number_of_classes)
            hidden_sizes_inverse_feature = (
                self.input_size + number_of_hidden_features,)

        for hidden_hidden_feature_index in range(number_of_hidden_features):
            self.hidden_feature_extractor_dict[hidden_hidden_feature_index] = MLPRegressor(
                hidden_sizes_feature,
                warm_start=True,
                verbose=True,
                early_stopping=True,
                max_iter=hidden_feature_max_iter,
                learning_rate_init=0.001,
                activation="tanh")

            self.inverse_hidden_feature_generator_dict[hidden_hidden_feature_index] = MLPRegressor(
                hidden_sizes_inverse_feature,
                warm_start=True,
                verbose=True,
                early_stopping=True,
                max_iter=inverse_feature_max_iter,
                learning_rate_init=0.001,
                activation="tanh")

        self.final_estimator = MLPRegressor(hidden_sizes_final,
                                            warm_start=True, verbose=True,
                                            early_stopping=True, max_iter=classification_max_iter,
                                            learning_rate_init=0.001, activation="tanh")

        self.shifted_signal_df = None
        self.shifted_signal_with_hidden_feature_df = None
        self.hidden_feature_df = None
        self.target_hidden_feature_vector_dict = None  # dict
        self.scalar_target_vector = None
        self.categorical_target_matrix = None
        self.signal_batch_size = signal_batch_size

    def time_shift_signal(self, X: pandas.DataFrame, save_shifted_train_signal_df=False, signal_batch_size=None):

        if signal_batch_size is None:
            signal_batch_size = self.signal_batch_size
        number_of_batches = math.ceil(len(X) / signal_batch_size)

        shifted_signal_df_by_batch_list = []
        for batch_index in range(number_of_batches):
            X_copy = X.iloc[
                     batch_index * signal_batch_size:(batch_index + 1) * signal_batch_size,
                     :].copy()
            for column_index in range(self.input_size):
                if "mean" in str(X_copy.columns[column_index]) or "std" in str(X_copy.columns[column_index]):
                    # Do not shift mean features
                    continue
                for step_index in range(-self.half_window_size, self.half_window_size + 1):
                    if step_index == 0:
                        continue
                    else:
                        X_copy[str(X_copy.columns[column_index]) + "_shift_" + str(step_index)] = X_copy[
                            X_copy.columns[column_index]].shift(
                            periods=step_index)
                        X_copy.fillna(method="ffill", inplace=True)
                        X_copy.fillna(method="bfill", inplace=True)
            shifted_signal_df_by_batch_list.append(X_copy)
        X_copy = pandas.concat(shifted_signal_df_by_batch_list, axis=0)
        sorted_column_list = sorted(list(X_copy.columns))
        X_copy = X_copy[sorted_column_list]
        if save_shifted_train_signal_df:
            self.shifted_signal_df = X_copy
        # print(X_copy)
        return X_copy

    def train_hidden_feature_extractor(self, X: pandas.DataFrame, y_scalar_vector, nb_iterations=1,
                                       reuse_shifted_signal_df=True):
        if self.target_hidden_feature_vector_dict is None:
            self.target_hidden_feature_vector_dict = {}
            for hidden_hidden_feature_index in range(self.number_of_hidden_features):
                self.target_hidden_feature_vector_dict[hidden_hidden_feature_index] = y_scalar_vector
        if reuse_shifted_signal_df and not self.shifted_signal_df is None:
            shifted_signal_df = self.shifted_signal_df
        else:
            shifted_signal_df = self.time_shift_signal(X, save_shifted_train_signal_df=True)
        print("TRAIN FEATURE EXTRACTORS")
        for iteration_index in range(nb_iterations):
            print("Feature Extractor - Iter:", iteration_index)
            for hidden_hidden_feature_index in range(self.number_of_hidden_features):
                print("Feature:", hidden_hidden_feature_index)
                target_feature_vector = (2 * y_scalar_vector + self.target_hidden_feature_vector_dict[
                    hidden_hidden_feature_index]) / 3
                fitting_result = self.hidden_feature_extractor_dict[hidden_hidden_feature_index].fit(shifted_signal_df,
                                                                                                     target_feature_vector
                                                                                                     )
                print("Feature Extractor:", hidden_hidden_feature_index, fitting_result)

    def compute_feature(self, X: pandas.DataFrame, reuse_shifted_signal_df=True, save_hidden_feature_df=True,
                        save_shifted_signal_with_hidden_feature_df=True):
        if reuse_shifted_signal_df and not self.shifted_signal_df is None:
            shifted_signal_df = self.shifted_signal_df
        else:
            shifted_signal_df = self.time_shift_signal(X, save_shifted_train_signal_df=True)

        shifted_signal_with_hidden_feature_df = shifted_signal_df.copy()
        hidden_feature_name_list = []
        for hidden_hidden_feature_index in range(self.number_of_hidden_features):
            shifted_signal_with_hidden_feature_df["hidden_feature_" + str(hidden_hidden_feature_index)] = \
                self.hidden_feature_extractor_dict[
                    hidden_hidden_feature_index].predict(shifted_signal_df)
            hidden_feature_name_list.append("hidden_feature_" + str(hidden_hidden_feature_index))

        self.shifted_signal_with_hidden_feature_df = shifted_signal_with_hidden_feature_df
        self.hidden_feature_df = shifted_signal_with_hidden_feature_df[hidden_feature_name_list]
        return shifted_signal_with_hidden_feature_df

    def train_final_estimator(self, X: pandas.DataFrame, y_scalar_vector, nb_iterations=1,
                              reuse_shifted_signal_df=True):

        shifted_signal_with_hidden_feature_df = self.compute_feature(X, reuse_shifted_signal_df)
        print("TRAIN FINAL ESTIMATOR")
        for iteration_index in range(nb_iterations):
            print("Final Estimator - Iter:", iteration_index)
            fiting_result = self.final_estimator.fit(shifted_signal_with_hidden_feature_df, y_scalar_vector)
            print("Final Estimator", fiting_result)

    def compute_final_output(self, X: pandas.DataFrame, reuse_shifted_signal_df=True):
        if reuse_shifted_signal_df and not self.shifted_signal_with_hidden_feature_df is None:
            shifted_signal_with_hidden_feature_df = self.shifted_signal_with_hidden_feature_df
        else:
            shifted_signal_with_hidden_feature_df = self.compute_feature(X, True)

        return self.final_estimator.predict(shifted_signal_with_hidden_feature_df)

    def train_inverse_hidden_feature_generator(self, X: pandas.DataFrame, y_scalar_vector, reuse_shifted_signal_df=True,
                                               reuse_hidden_feature_df=True, nb_iterations=1, ):
        input_for_inverse_hidden_feature_generator_df = self.shifted_signal_df.copy()
        final_estimator_output = self.compute_final_output(X, True)
        input_for_inverse_hidden_feature_generator_df["Final"] = (final_estimator_output + 2 * y_scalar_vector) / 3
        print("TRAIN INVERSE FEATURE GENERATOR")
        for hidden_hidden_feature_index in range(self.number_of_hidden_features):
            print("Feature:", hidden_hidden_feature_index)
            fitting_result = self.inverse_hidden_feature_generator_dict[hidden_hidden_feature_index].fit(
                input_for_inverse_hidden_feature_generator_df,
                self.hidden_feature_df["hidden_feature_" + str(hidden_hidden_feature_index)])
            print("INVERSE FEATURE:", hidden_hidden_feature_index, fitting_result)
        pass

    def compute_target_for_hidden_feature_extractor_training(self, y_scalar_vector, build_final_estimator_input=False):
        print("compute_target_for_hidden_feature_extractor_training")
        input_for_inverse_hidden_feature_generator_df = self.shifted_signal_df.copy()
        input_for_inverse_hidden_feature_generator_df["Final"] = (y_scalar_vector)
        self.target_hidden_feature_vector_dict = {}

        for hidden_hidden_feature_index in range(self.number_of_hidden_features):
            hidden_feature_vector_tmp = self.inverse_hidden_feature_generator_dict[hidden_hidden_feature_index].predict(
                input_for_inverse_hidden_feature_generator_df)

            self.target_hidden_feature_vector_dict[hidden_hidden_feature_index] = hidden_feature_vector_tmp
        # print("Alive")
        if build_final_estimator_input:
            input_for_final_estimator_df = self.shifted_signal_df.copy()
            # print(input_for_final_estimator_df.columns)
            for hidden_hidden_feature_index in range(self.number_of_hidden_features):
                # print(self.target_hidden_feature_vector_dict[hidden_hidden_feature_index])

                input_for_final_estimator_df["hidden_feature_"
                                             + str(hidden_hidden_feature_index)] = \
                    self.target_hidden_feature_vector_dict[hidden_hidden_feature_index]
            # input_for_final_estimator_df["Final"] = y_scalar_vector
            return input_for_final_estimator_df

    def train_network(self, X: pandas.DataFrame, y_scalar_vector, nb_steps=10):
        for step_index in range(nb_steps):
            start_ = time.time()
            print("STEP:", step_index)
            self.train_hidden_feature_extractor(X, y_scalar_vector)
            self.train_final_estimator(X, y_scalar_vector)
            prediction_this_step = self.compute_final_output(X)
            self.train_inverse_hidden_feature_generator(X, y_scalar_vector)
            ideal_input_for_final_estimator_df = self.compute_target_for_hidden_feature_extractor_training(
                y_scalar_vector,
                build_final_estimator_input=True)

            prediction_next_step = self.final_estimator.predict(ideal_input_for_final_estimator_df)

            print("MSE this step:", metrics.mean_squared_error(y_scalar_vector, prediction_this_step))
            print("Expected MSE next step:", metrics.mean_squared_error(y_scalar_vector, prediction_next_step))
            print("END STEP:", step_index, "in", str(time.time() - start_))
            now_time = time.time()
            pickle.dump(self.final_estimator, open("file_estimator_" + str(now_time) + ".pickle", mode="wb"))
            pickle.dump(self.hidden_feature_extractor_dict,
                        open("hidden_feature_extractor_dict_" + str(now_time) + ".pickle", mode="wb"))
            pickle.dump(self.inverse_hidden_feature_generator_dict,
                        open("inverse_hidden_feature_generator_dict_" + str(now_time) + ".pickle", mode="wb"))

            json.dump(list(self.shifted_signal_with_hidden_feature_df.columns),
                      open("description_" + str(now_time) + ".json", mode="wb"))


print("Load data")
df = pandas.read_csv("train.csv")
# df = df.iloc[4000000:5000000, :]
df = df.iloc[000000:5000000, :]

# df["mean_100"] = df["signal"].rolling(window=100, center=True).mean()

windows = [100, 200, 300, 500, 1000]
for window_ in windows:
    df["mean_" + str(window_)] = df["signal"].rolling(window=window_, center=True).mean()
    df["std_" + str(window_)] = df["signal"].rolling(window=window_, center=True).std()
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
input_feature_list = ["signal"]
print("Go")
for window_ in windows:
    input_feature_list.append("mean_" + str(window_))
train_input_df = df[input_feature_list]
# train_input_df = df[["signal_minus_mean_100"]]
train_output_df = df["open_channels"]

input_size = len(train_input_df.columns)
encoder = LabelEncoder()
train_output_df = encoder.fit_transform(train_output_df)
number_of_classes = len(set(list(numpy.array(train_output_df))))

categorical_output = numpy.zeros((len(train_output_df), number_of_classes), dtype=float)

categorical_output[numpy.arange(len(train_output_df)), train_output_df] = 1
probability_vector = categorical_output.mean(axis=0)
###
model_ = CustomMLPClassifier(input_size, number_of_classes, number_of_hidden_features=number_of_classes,
                             half_window_size=10,
                             signal_batch_size=500000, simple_test=False)
'''
print(model_.time_shift_signal(train_input_df))
model_.train_hidden_feature_extractor(train_input_df, train_output_df)
# print(model_.compute_feature(train_input_df))
model_.train_final_estimator(train_input_df, train_output_df)
model_.train_inverse_hidden_feature_generator(train_input_df, train_output_df)
input_for_final_estimator = model_.compute_target_for_hidden_feature_extractor_training(train_output_df,
                                                                                 build_final_estimator_input=True)
print("input_for_final_estimator", input_for_final_estimator)

prediction_before = model_.compute_final_output(train_input_df)
prediction_after = model_.final_estimator.predict(input_for_final_estimator)
print(metrics.mean_squared_error(train_output_df, prediction_before))
print(metrics.mean_squared_error(train_output_df, prediction_after))

'''
model_.train_network(train_input_df, train_output_df, nb_steps=20)
