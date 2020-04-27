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

mean_correlation_dict_list = []
base_window_size = 250
slice_length = 500000
for slice_index in range(0, 10):
    feature_df = TRAIN_DF.loc[slice_length * (slice_index):slice_length * (slice_index + 1) - 1, :].copy()
    feature_df["signal_recurrent_mean"] = feature_df["signal"]
    for recurrent_id in range(1, 21):
        # print(feature_df)
        feature_df["signal_recurrent_mean"] = feature_df["signal_recurrent_mean"].rolling(center=True,
                                                                                          window=base_window_size).mean()
        # print(feature_df)
        # print(base_window_size * window_scale)
        feature_df_temp = feature_df[["signal", "signal_recurrent_mean", "open_channels"]].copy()
        feature_df_temp["signal_minus_recurrent_mean"] = feature_df_temp["signal"] - feature_df_temp[
            "signal_recurrent_mean"]
        feature_df_temp.dropna(inplace=True)
        corr_coeff_ = numpy.corrcoef(feature_df_temp["signal_minus_recurrent_mean"], feature_df_temp["open_channels"])[0][1]
        print(slice_index, recurrent_id * base_window_size, corr_coeff_)
        mean_correlation_dict_list.append(
            {"batch": slice_index, "window": recurrent_id * base_window_size, "correlation": corr_coeff_})

mean_correlation_df = pandas.DataFrame(data=mean_correlation_dict_list)
mean_correlation_df = mean_correlation_df[["batch", "window", "correlation"]]
mean_correlation_df.to_csv("signal_minus_recurrent_mean_correlation.csv", index=False)
print(mean_correlation_df)
