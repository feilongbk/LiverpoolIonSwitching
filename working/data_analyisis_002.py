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

#### CORRELATION WITH PAST ####
window_size = 50

slice_ = 0

TRAIN_FEATURES_DF = TRAIN_DF.loc[500000 * (slice_):500000 * (slice_ + 1) - 1, :].copy()
# TRAIN_FEATURES_DF = TRAIN_DF.copy()
TRAIN_FEATURES_DF["signal_mean"] = TRAIN_FEATURES_DF["signal"].rolling(center=True, window=window_size).mean()
TRAIN_FEATURES_DF["signal_mean_std"] = TRAIN_FEATURES_DF["signal_mean"].rolling(center=True, window=window_size).std()
TRAIN_FEATURES_DF["signal_mean_mean"] = TRAIN_FEATURES_DF["signal_mean"].rolling(center=True, window=window_size).mean()
TRAIN_FEATURES_DF["signal_mean_removing_mean_mean"] = TRAIN_FEATURES_DF["signal_mean"] - TRAIN_FEATURES_DF[
    "signal_mean_mean"]
TRAIN_FEATURES_DF["signal_std"] = TRAIN_FEATURES_DF["signal"].rolling(center=True, window=window_size).std()
TRAIN_FEATURES_DF["signal_per_std"] = TRAIN_FEATURES_DF["signal"] / TRAIN_FEATURES_DF["signal_std"]
TRAIN_FEATURES_DF["signal_var"] = TRAIN_FEATURES_DF["signal"].rolling(center=True, window=window_size).var()
TRAIN_FEATURES_DF["signal_removing_mean"] = TRAIN_FEATURES_DF["signal"] - TRAIN_FEATURES_DF["signal_mean"]
TRAIN_FEATURES_DF["signal_removing_mean_mean"] = TRAIN_FEATURES_DF["signal"] - TRAIN_FEATURES_DF["signal_mean_mean"]
TRAIN_FEATURES_DF["signal_removing_mean_per_std"] = TRAIN_FEATURES_DF["signal_removing_mean"] / TRAIN_FEATURES_DF[
    "signal_std"]

# scipy.signal.windows.gaussian
print(TRAIN_FEATURES_DF)

TRAIN_FEATURES_DF.dropna(inplace=True)
feature_list = ["signal_mean_std","signal_per_std", "signal", "signal_mean", "signal_mean_mean", "signal_std", "signal_var",
                "signal_removing_mean",
                "signal_removing_mean_per_std", "signal_mean_removing_mean_mean", "signal_removing_mean_mean"]
corr_coeff_dict = {}
for feature_ in feature_list:
    corr_coeff_dict[feature_] = numpy.corrcoef(TRAIN_FEATURES_DF[feature_], TRAIN_FEATURES_DF["open_channels"])
    print(feature_, corr_coeff_dict[feature_][0][1])

# pyplot.scatter(TRAIN_FEATURES_DF["signal_mean"], TRAIN_FEATURES_DF["open_channels"], marker='.')

# Calculate the point density
# xy = numpy.vstack([TRAIN_FEATURES_DF["open_channels"], TRAIN_FEATURES_DF["signal_mean"]])
# z = gaussian_kde(xy)(xy)
bins = (10, 100)
#feature_vector = TRAIN_FEATURES_DF["signal"]
# feature_vector = TRAIN_FEATURES_DF["signal_mean"]
feature_vector = TRAIN_FEATURES_DF["signal_std"]
# feature_vector = TRAIN_FEATURES_DF["signal_removing_mean"]
# feature_vector = TRAIN_FEATURES_DF["signal_removing_mean_mean"]
pyplot.subplot(1, 4, 1)
pyplot.hist2d(feature_vector,TRAIN_FEATURES_DF["open_channels"],  bins, density=True, cmap=pyplot.cm.Reds)
# pyplot.hist2d(TRAIN_FEATURES_DF["open_channels"], TRAIN_FEATURES_DF["signal_std"], bins, density=True,cmap=pyplot.cm.Reds)
# pyplot.hist2d(TRAIN_FEATURES_DF["open_channels"], TRAIN_FEATURES_DF["signal"], bins, density=True,cmap=pyplot.cm.Reds)
# pyplot.hist2d(TRAIN_FEATURES_DF["open_channels"], TRAIN_FEATURES_DF["signal_removing_mean"], bins, density=True,cmap=pyplot.cm.Reds)
# pyplot.hist2d(TRAIN_FEATURES_DF["open_channels"], TRAIN_FEATURES_DF["signal_removing_mean_per_std"], bins, density=True,cmap=pyplot.cm.Reds)
# pyplot.hist2d(TRAIN_FEATURES_DF["open_channels"], TRAIN_FEATURES_DF["signal_var"], bins, density=True,cmap=pyplot.cm.Reds)
pyplot.colorbar()
pyplot.subplot(1, 4, 2)
pyplot.scatter(feature_vector,TRAIN_FEATURES_DF["open_channels"])

pyplot.subplot(1, 4, 3)
pyplot.hist(TRAIN_FEATURES_DF["open_channels"], density=True)

pyplot.subplot(1, 4, 4)
pyplot.hist(feature_vector, density=True)

print(TRAIN_FEATURES_DF.describe()["open_channels"])
print(feature_vector.describe())
print(linregress(feature_vector, TRAIN_FEATURES_DF["open_channels"]))
# fig, ax = pyplot.subplots()
# ax.scatter(TRAIN_FEATURES_DF["open_channels"], TRAIN_FEATURES_DF["signal_mean"], c=z, s=100, edgecolor='')
pyplot.show()
