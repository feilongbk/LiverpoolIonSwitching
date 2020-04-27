import pandas
import scipy
from scipy.special import softmax
from scipy.optimize import minimize, least_squares, root, basinhopping
from sklearn.neural_network import MLPClassifier
import numpy
import time
from sklearn.metrics import log_loss


#### Connection_Graph
#### Input_layer ####
# Input_feature_partition=list(int) of which sum = input layer size
# ie
#### Output_layer ####


class CustomMLPClassifier001():
    def __init__(self, number_of_hidden_features, semi_window_size, input_size, number_of_classes):
        self.mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(number_of_hidden_features * number_of_classes), warm_start=True, verbose=True,
            early_stopping=True, max_iter=1, learning_rate_init=0.001)
        self.number_of_hidden_features = number_of_hidden_features
        self.semi_window_size = semi_window_size
        self.input_size = input_size
        self.feature_transform_tensor = numpy.zeros(
            (number_of_hidden_features, 2 * semi_window_size + 1, input_size)) / (
                                                number_of_hidden_features * (2 * semi_window_size + 1) * input_size)

        for hidden_feature_index in range(self.number_of_hidden_features):
            for input_index in range(self.input_size):
                self.feature_transform_tensor[hidden_feature_index][self.semi_window_size][
                    input_index] = 1.0 / number_of_hidden_features
        self.feature_transform_offset_vector = numpy.zeros(number_of_hidden_features)
        ####
        self.number_of_classes = number_of_classes
        pass

    def compute_feature(self, time_series_of_features: pandas.DataFrame) -> pandas.DataFrame:
        ## run by hidden feature
        X = time_series_of_features
        hidden_feature_col_dict = {}
        for hidden_feature_index in range(self.number_of_hidden_features):
            feature_col = 0
            for column_index in range(self.input_size):
                for step_index in range(-self.semi_window_size, self.semi_window_size + 1):
                    if step_index == 0:
                        feature_contribution_col = X[X.columns[column_index]]
                    else:
                        feature_contribution_col = X[X.columns[column_index]].shift(periods=step_index)
                        feature_contribution_col.fillna(method="bfill", inplace=True)
                        feature_contribution_col.fillna(method="ffill", inplace=True)
                    feature_contribution_col = \
                        self.feature_transform_tensor[hidden_feature_index][self.semi_window_size + step_index][
                            column_index] * feature_contribution_col + self.feature_transform_offset_vector[
                            hidden_feature_index]
                    feature_col += feature_contribution_col
            hidden_feature_col_dict[hidden_feature_index] = feature_col

        hidden_feature_col_df = pandas.DataFrame(hidden_feature_col_dict)
        return hidden_feature_col_df

    def predict_from_features(self, hidden_feature_col_df: pandas.DataFrame):
        return self.mlp_classifier.predict_proba(hidden_feature_col_df)

    def predict_proba(self, time_series_of_features: pandas.DataFrame) -> pandas.DataFrame:
        hidden_feature_col_df = self.compute_feature(time_series_of_features)
        output_probability_array = self.predict_from_features(hidden_feature_col_df)
        return output_probability_array

    def parameters_to_weight_tensors(self, parameters_vector) -> dict:
        result = {}
        index_temp = 0

        ### feature_transform_tensor
        feature_transform_tensor_flatten_size = numpy.prod((self.feature_transform_tensor).shape)
        result["feature_transform_tensor"] = numpy.reshape(
            parameters_vector[index_temp:index_temp + feature_transform_tensor_flatten_size],
            newshape=(self.feature_transform_tensor).shape)
        index_temp += feature_transform_tensor_flatten_size

        ### feature_transform_offset_vector
        result["feature_transform_offset_vector"] = parameters_vector[
                                                    index_temp:index_temp + self.number_of_hidden_features]

        index_temp += self.number_of_hidden_features

        return result

    ###

    def fit_step(self, time_series_of_features: pandas.DataFrame, target_output):
        print("Train neural network MLPClassifier")
        ### Fit MLP classifier
        t0_hidden_feature_df = self.compute_feature(time_series_of_features)
        self.mlp_classifier.fit(t0_hidden_feature_df, target_output)
        ### Fit

        iteration_index = {"value": 0}

        def loss_function_on_proba(parameters):
            print("Iteration:", iteration_index["value"])
            temp_classifier = CustomMLPClassifier001(self.number_of_hidden_features, self.semi_window_size,
                                                     self.input_size,
                                                     self.number_of_classes)
            tensor_dict = self.parameters_to_weight_tensors(parameters_vector=parameters)
            print("Parameters:", tensor_dict)
            temp_classifier.__setattr__("feature_transform_tensor", tensor_dict["feature_transform_tensor"])
            temp_classifier.__setattr__("feature_transform_offset_vector",
                                        tensor_dict["feature_transform_offset_vector"])

            temp_hidden_feature_df = self.compute_feature(time_series_of_features)

            prediction = self.mlp_classifier.predict_proba(temp_hidden_feature_df)
            # print(prediction)
            # loss = numpy.mean(numpy.square(numpy.array(prediction) - numpy.array(target_output)))
            loss = log_loss(numpy.array(target_output), numpy.array(prediction))
            print("Loss:", loss)
            iteration_index["value"] = iteration_index["value"] + 1
            return loss

        x_0 = []
        x_0.extend(
            numpy.reshape(self.feature_transform_tensor, newshape=numpy.prod(self.feature_transform_tensor.shape)))
        x_0.extend(self.feature_transform_offset_vector)
        x_0 = numpy.array(x_0)
        #### Boundary Condition
        bnds_1 = []
        for i in range(len(x_0)):
            bnds_1.append([-1, 1])
        bnds_2 = (x_0 * 0.0 - 1, x_0 * 0.0 + 1)
        print("Train feature extractor")
        result = minimize(loss_function_on_proba, x_0)
        feature_tensor_dict = self.parameters_to_weight_tensors(parameters_vector=result.x)
        print(result)
        print(feature_tensor_dict)
        self.__setattr__("feature_transform_tensor", feature_tensor_dict["feature_transform_tensor"])
        self.__setattr__("feature_transform_offset_vector",
                         feature_tensor_dict["feature_transform_offset_vector"])

    def fit(self, time_series_of_features: pandas.DataFrame, target_output, number_of_steps=100):
        for step_index in range(number_of_steps):
            print("Step:", step_index)
            self.fit_step(time_series_of_features, target_output, )


### test


df = pandas.read_csv("train.csv")
print(df.columns)
df["mean_100"] = df["signal"].rolling(window=100, center=True).mean()
df.fillna(method="bfill", inplace=True)
df.fillna(method="ffill", inplace=True)
df["signal_minus_mean_100"] = df["signal"] - df["mean_100"]
# train_input_df = df[["signal"]]
train_input_df = df[["signal", "mean_100"]]
# train_input_df = df[["signal_minus_mean_100"]]
train_output_df = df["open_channels"]
# print(numpy.array(train_output_df))


number_of_classes = len(set(list(numpy.array(train_output_df))))
categorical_output = numpy.zeros((len(train_output_df), number_of_classes), dtype=float)

categorical_output[numpy.arange(len(train_output_df)), train_output_df] = 1
probability_vector = categorical_output.mean(axis=0)
probability_dict = {}
for i in range(len(probability_vector)):
    probability_dict[i] = probability_vector[i]
print()
classifier_ = CustomMLPClassifier001(number_of_hidden_features=number_of_classes, semi_window_size=10,
                                     input_size=len(train_input_df.columns),
                                     number_of_classes=number_of_classes)
start_ = time.time()
classifier_.fit(train_input_df, categorical_output)
# output_ = classifier_.predict_proba(train_input_df)
print(time.time() - start_)
