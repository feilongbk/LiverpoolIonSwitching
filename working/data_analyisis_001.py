import numpy
import pandas
from matplotlib import pyplot
import math
import os
import scipy

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
print(DIR_NAME)

TRAIN_DF = pandas.read_csv(DIR_NAME + "/train.csv")
TEST_DF = pandas.read_csv(DIR_NAME + "/test.csv")

#### Investigate the correlation of the TARGET (open_channels) to different differential features
DELTA_T = 0.0001
TRAIN_FEATURES_DF = TRAIN_DF.copy()
### Differentiating
TRAIN_FEATURES_DF["signal_d0"] = TRAIN_FEATURES_DF["signal"]
for i in range(2):
    TRAIN_FEATURES_DF["signal_d" + str(i + 1)] = TRAIN_FEATURES_DF["signal_d" + str(i)].diff(periods=1) / DELTA_T

print(TRAIN_FEATURES_DF)
TRAIN_FEATURES_DF.dropna(inplace=True)
corr_coeff_dict = {}
for i in range(3):
    corr_coeff_dict[i] = numpy.corrcoef(TRAIN_FEATURES_DF["signal_d" + str(i)], TRAIN_FEATURES_DF["open_channels"])
    print(corr_coeff_dict[i])

#### CORRELATION WITH TIME ####

time_correlation_dict = {}

STEP = 1
MAX_PERIOD = 50

#### CORRELATION WITH PAST ####

TRAIN_FEATURES_DF = TRAIN_DF.copy()
TRAIN_FEATURES_DF["signal_pN"] = TRAIN_FEATURES_DF["signal"]

corr_coeff_dict = {}
for i in range(MAX_PERIOD):
    TRAIN_FEATURES_DF.dropna(inplace=True)
    corr_coeff_dict[i] = numpy.corrcoef(TRAIN_FEATURES_DF["signal_pN"], TRAIN_FEATURES_DF["open_channels"])
    print(corr_coeff_dict[i])
    time_correlation_dict[-i] = corr_coeff_dict[i][0][1]
    TRAIN_FEATURES_DF["signal_pN"] = TRAIN_FEATURES_DF["signal_pN"].shift(periods=STEP)

#### CORRELATION WITH FUTURE ####
TRAIN_FEATURES_DF = TRAIN_DF.copy()
TRAIN_FEATURES_DF["signal_pN"] = TRAIN_FEATURES_DF["signal"]

corr_coeff_dict = {}
for i in range(MAX_PERIOD):
    TRAIN_FEATURES_DF.dropna(inplace=True)
    corr_coeff_dict[i] = numpy.corrcoef(TRAIN_FEATURES_DF["signal_pN"], TRAIN_FEATURES_DF["open_channels"])
    print(corr_coeff_dict[i])
    time_correlation_dict[i] = corr_coeff_dict[i][0][1]
    TRAIN_FEATURES_DF["signal_pN"] = TRAIN_FEATURES_DF["signal_pN"].shift(periods=STEP)

print(time_correlation_dict)

x_ = list(time_correlation_dict.keys())
y_ = []
for x_value in x_:
    y_.append(time_correlation_dict[x_value])

pyplot.scatter(x_, y_)
pyplot.show()
