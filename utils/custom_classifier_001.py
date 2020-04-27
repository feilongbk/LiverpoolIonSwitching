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
    def __init__(self, number_of_hidden_features, semi_window_size, input_size, number_of_classes, prior_probability):
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
        self.prior_probability = prior_probability

        self.hidden_feature_to_output_logits_tensor = numpy.ones((number_of_classes, number_of_hidden_features))
        self.hidden_feature_to_output_logits_tensor = self.hidden_feature_to_output_logits_tensor / number_of_hidden_features / number_of_classes
        self.hidden_feature_to_output_logits_offset_vector = numpy.zeros(number_of_classes)
        '''
                for output_logit_index in range(0, number_of_classes):
            self.hidden_feature_to_output_logits_tensor[output_logit_index] = self.prior_probability[
                                                                                  output_logit_index] * \
                                                                              self.hidden_feature_to_output_logits_tensor[
                                                                                  output_logit_index]
        '''

        self.final_activation = "identity"
        pass

    def compute(self, time_series_of_features: pandas.DataFrame) -> pandas.DataFrame:
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

        output_logits_dict = {}
        #### Hidden feature to output logits
        for output_logit_index in range(self.number_of_classes):
            logit_col = 0
            for hidden_feature_index in range(self.number_of_hidden_features):
                logit_col += hidden_feature_col_df[hidden_feature_index] * \
                             self.hidden_feature_to_output_logits_tensor[output_logit_index][hidden_feature_index] + \
                             self.hidden_feature_to_output_logits_offset_vector[output_logit_index]

            output_logits_dict[output_logit_index] = logit_col

        output_logits_df = pandas.DataFrame(output_logits_dict)
        # print(hidden_feature_col_df)
        # print(output_logits_df)
        # class_probability_df = softmax(output_logits_df, axis=0)
        output_logits_df = 1 / (1 + numpy.exp(-output_logits_df))
        '''

        
        '''
        # soft_max
        logit_exp_sum = 0
        output_probability_dict = {}
        for output_logit_index in range(self.number_of_classes):
            output_probability_dict[output_logit_index] = numpy.exp(output_logits_df[output_logit_index])
            logit_exp_sum += output_probability_dict[output_logit_index]

        inverted_logit_exp_sum = 1 / logit_exp_sum
        output_probability_df = pandas.DataFrame(output_probability_dict)

        for output_logit_index in range(self.number_of_classes):
            output_probability_df[output_logit_index] = output_probability_df[
                                                            output_logit_index] * inverted_logit_exp_sum

        return output_probability_df

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
        ### hidden_feature_to_output_logits_tensor
        hidden_feature_to_output_logits_tensor_flatten_size = numpy.prod(
            (self.hidden_feature_to_output_logits_tensor).shape)
        result["hidden_feature_to_output_logits_tensor"] = numpy.reshape(
            parameters_vector[index_temp:index_temp + hidden_feature_to_output_logits_tensor_flatten_size],
            newshape=(self.hidden_feature_to_output_logits_tensor).shape)

        index_temp += hidden_feature_to_output_logits_tensor_flatten_size
        ### feature_transform_offset_vector
        result["hidden_feature_to_output_logits_offset_vector"] = parameters_vector[
                                                                  index_temp:index_temp + self.number_of_classes]

        return result

    ###

    def fit(self, time_series_of_features: pandas.DataFrame, target_output):
        iteration_index = {"value": 0}

        def loss_function_on_proba(parameters):
            print("Iteration:", iteration_index["value"])
            temp_classifier = CustomMLPClassifier001(self.number_of_hidden_features, self.semi_window_size,
                                                     self.input_size,
                                                     self.number_of_classes, self.prior_probability)
            tensor_dict = self.parameters_to_weight_tensors(parameters_vector=parameters)
            print("Parameters:", tensor_dict)
            temp_classifier.__setattr__("feature_transform_tensor", tensor_dict["feature_transform_tensor"])
            temp_classifier.__setattr__("feature_transform_offset_vector",
                                        tensor_dict["feature_transform_offset_vector"])

            temp_classifier.__setattr__("hidden_feature_to_output_logits_tensor",
                                        tensor_dict["hidden_feature_to_output_logits_tensor"])
            temp_classifier.__setattr__("hidden_feature_to_output_logits_offset_vector",
                                        tensor_dict["hidden_feature_to_output_logits_offset_vector"])
            prediction = temp_classifier.compute(time_series_of_features)
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
        x_0.extend(numpy.reshape(self.hidden_feature_to_output_logits_tensor,
                                 newshape=numpy.prod(self.hidden_feature_to_output_logits_tensor.shape)))
        x_0.extend(self.hidden_feature_to_output_logits_offset_vector)
        x_0 = numpy.array(x_0)

        #### Boundary Condition
        '''
                cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
                ...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
                ...{'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
        '''
        bnds_1 = []
        for i in range(len(x_0)):
            bnds_1.append([-1, 1])
        bnds_2 = (x_0 * 0.0 - 1, x_0 * 0.0 + 1)
        # print(x_0)
        # least_squares(loss_function_on_proba, x_0)

        # result = minimize(loss_function_on_proba, x_0, method="Powell", bounds=bnds)  ## OK
        # result = minimize(loss_function_on_proba, x_0, method="lm", bounds=bnds)  ## OK
        # result = root(loss_function_on_proba, x_0, method="lm", bounds=bnds)  ## OK
        # result = root(loss_function_on_proba, x_0, method="lm")  ## OK
        # result = minimize(loss_function_on_proba, x_0, method="L-BFGS-B", bounds=bnds)  ## OK
        # result = least_squares(loss_function_on_proba, x_0, method="lm", bounds=bnds)  ## OK

        #### 1 use Powell unconstrained
        # result = minimize(loss_function_on_proba, x_0, method="Powell", options={'maxfun': 10, 'maxiter': 10})
        #result = minimize(loss_function_on_proba, x_0, bounds=bnds_1, options={'maxfun': 10, 'maxiter': 10})
        result = basinhopping(loss_function_on_proba, x_0,)
        print(result)

        print(result)


### test


df = pandas.read_csv("train.csv")
print(df.columns)
df["mean_100"] = df["signal"].rolling(window=100, center=True).mean()
df.fillna(method="bfill", inplace=True)
df.fillna(method="ffill", inplace=True)
df["signal_minus_mean_100"] = df["signal"] - df["mean_100"]
# train_input_df = df[["signal"]]
train_input_df = df[["signal_minus_mean_100"]]
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
classifier_ = CustomMLPClassifier001(number_of_hidden_features=number_of_classes, semi_window_size=1,
                                     input_size=len(train_input_df.columns),
                                     number_of_classes=number_of_classes, prior_probability=probability_dict)
start_ = time.time()
classifier_.fit(train_input_df, categorical_output)
# output_ = classifier_.compute(train_input_df)
print(time.time() - start_)
# print(output_)
# print(classifier_.compute(train_df))
# print(softmax(train_df,axis=1))
