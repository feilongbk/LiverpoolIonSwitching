import numpy
import pandas
from matplotlib import pyplot
import math
import os
import scipy
from scipy.stats import gaussian_kde, linregress

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
print(DIR_NAME)

TRAIN_DF = pandas.read_csv(DIR_NAME + "/train.csv")
TEST_DF = pandas.read_csv(DIR_NAME + "/test.csv")

#### CORRELATION WITH TIME ####

time_correlation_dict = {}

STEP = 1
MAX_PERIOD = 50

#### CORRELATION WITH STATISTICS ####

#### Standard Deviation ####

std_correlation_dict_list = []
base_window_size = 10
slice_length = 500000
for slice_index in range(0, 10):
    for window_scale in range(1, 51):
        feature_df = TRAIN_DF.loc[slice_length * (slice_index):slice_length * (slice_index + 1) - 1, :].copy()
        # print(feature_df)
        feature_df["signal_std"] = feature_df["signal"].rolling(center=True,
                                                                window=base_window_size * window_scale).std()
        # print(feature_df)
        # print(base_window_size * window_scale)
        feature_vector = feature_df["signal_std"]
        feature_df.dropna(inplace=True)
        corr_coeff_ = numpy.corrcoef(feature_df["signal_std"], feature_df["open_channels"])[0][1]
        print(slice_index, window_scale * base_window_size, corr_coeff_)
        std_correlation_dict_list.append(
            {"batch": slice_index, "window": window_scale * base_window_size, "correlation": corr_coeff_})

std_correlation_df = pandas.DataFrame(data=std_correlation_dict_list)
std_correlation_df = std_correlation_df[["batch", "window", "correlation"]]
std_correlation_df.to_csv("std_correlation.csv", index=False)
print(std_correlation_df)
