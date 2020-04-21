import numpy
import pandas
import matplotlib
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

#### CORRELATION OVER TIME ####
STEP = 1
MAX_PERIOD = 20
TRAIN_FEATURES_DF = TRAIN_DF.copy()
TRAIN_FEATURES_DF["signal_p0"] = TRAIN_FEATURES_DF["signal"]
for i in range(MAX_PERIOD):
    TRAIN_FEATURES_DF["signal_p" + str(i + 1)] = TRAIN_FEATURES_DF["signal_p" + str(i)].shift(periods=STEP)

print(TRAIN_FEATURES_DF)
TRAIN_FEATURES_DF.dropna(inplace=True)
corr_coeff_dict = {}
for i in range(MAX_PERIOD):
    corr_coeff_dict[i] = numpy.corrcoef(TRAIN_FEATURES_DF["signal_p" + str(i)], TRAIN_FEATURES_DF["open_channels"])
    print(corr_coeff_dict[i])
