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

signal_correlation_dict_list = []
base_window_size = 10
slice_length = 500000
for slice_index in range(0, 10):
    for window_scale in range(1, 2):
        feature_df = TRAIN_DF.loc[slice_length * (slice_index):slice_length * (slice_index + 1) - 1, :].copy()
        # print(feature_df)
        feature_df["signal"] = feature_df["signal"]
        # print(feature_df)
        # print(base_window_size * window_scale)
        feature_vector = feature_df["signal"]
        feature_df.dropna(inplace=True)
        corr_coeff_ = numpy.corrcoef(feature_df["signal"], feature_df["open_channels"])[0][1]
        print(slice_index, window_scale * base_window_size, corr_coeff_)
        signal_correlation_dict_list.append(
            {"batch": slice_index, "window": window_scale * base_window_size, "correlation": corr_coeff_})

signal_correlation_df = pandas.DataFrame(data=signal_correlation_dict_list)
signal_correlation_df = signal_correlation_df[["batch", "window", "correlation"]]
signal_correlation_df.to_csv("signal_correlation.csv", index=False)
print(signal_correlation_df)