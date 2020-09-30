import sys
from ipywidgets import interact
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from download import download
import numpy as np

pd.set_option('display.max_rows', 50000)
df_bikes = pd.read_csv("bicycle_db.csv",
                       na_values="", converters={'data': str, 'heure': str})
df_bikes['heure'] = df_bikes['heure'].replace('', np.nan)
df_bikes.iloc[400:402]
df_bikes.dropna(subset=['heure'], inplace=True)
df_bikes.iloc[399:402]
df_bikes['date'] + ' ' + df_bikes['heure'] + ':00'
# ADAPT OLD to create the df_bikes['Time']

time_improved = pd.to_datetime(df_bikes['date'] +
                               ' ' + df_bikes['heure'] + ':00',
                               format='%Y-%m-%d %H:%M')

# Where d = day, m=month, Y=year, H=hour, M=minutes
# create correct timing format in the dataframe

df_bikes['Time'] = time_improved
df_bikes.set_index('Time', inplace=True)
# remove useles columns
del df_bikes['heure']
del df_bikes['date']
accidents_week = df_bikes.sort_values(['Time'])

# The input, equal to the complete data set :
input = accidents_week
# The output, composed by the data collected in 2017 :
output = accidents_week.iloc[58561:63389:, :]
accidents_week = input.groupby(['weekday', input.index.hour])[
    'sexe'].count().unstack(level=0)
accidents_d = accidents_week.sum()
# Number of accident by days of week (average over the year) :
print(accidents_d/14)
accidents_d = accidents_d/14
accidents_week_2017 = output.groupby(['weekday', output.index.hour])[
    'sexe'].count().unstack(level=0)
accidents_d_2017 = accidents_week_2017.sum()
# Number of accident by days of week in 2017 :
print(accidents_d_2017)
accident_2017 = accidents_d_2017

np.set_printoptions(threshold=sys.maxsize)
width = 6
raw_x = np.array(accidents_d)
raw_y = np.array(accident_2017)
# Plot of the input and the output
fig = plt.figure(figsize=(7, 7))
ax = plt.axes()
ax.plot(raw_x, "r.-", label="input")
ax.plot(raw_y, "b.-", label="output")
ax.set_xlabel("Jour de la semaine")
ax.set_ylabel("Accidents")
ax.legend()
plt.title(
    "Moyenne du nombe d'accidents par jour de la semaine (Lundi = 0, ..., Dimanche=6)")
plt.grid()

T = len(raw_x)
A = np.zeros((T, width))
A[0:, 0] = raw_x[0:]
for i in range(1, width):
    # We create a matrix to solve Y=AW, with W being the coefficient matrix
    A[i:, i] = raw_x[0:-i]

# Output of A^-1Y=W
wop = np.dot(np.linalg.pinv(A), raw_y)
# Thanks to the estimation of the coefficients, we can estimate Y :
y_pred = np.dot(A, wop)

# comparison of the real Y and the predicted Y. With width = 6, the curves are almost superposed
fig = plt.figure(figsize=(7, 7))
ax = plt.axes()
ax.plot(accident_2017, "r.-", label="real_output")
ax.plot(y_pred, "b.-", label="prediction")
ax.set_xlabel("Jour de la semaine")
ax.set_ylabel("Accidents")
ax.legend()
plt.grid()
plt.title("Moyenne du nombe d'accidents par jour de la semaine en 2017 (Lundi = 0, ..., Dimanche=6)")

# Calculation of the error for different values of witdh.
err = []
max_window = 200
raw_x = np.array(np.array(accidents_d))
raw_y = np.array(np.array(accident_2017))
T = len(raw_x)
for width in range(1, max_window+1):
    A = np.zeros((T, width))
    A[0:, 0] = raw_x[0:]
    for i in range(1, width):
        A[i:, i] = raw_x[0:-i]
    wop = np.dot(np.linalg.pinv(A), raw_y)
    y_pred = np.dot(A, wop)
    err.append(np.linalg.norm(y_pred-raw_y))

# Plot of the errors : with width >= 6, the error is near from 0 so the prediction is very close to the real value
fig = plt.figure(figsize=(16, 9))
ax = plt.axes()
ax.plot(err, ".-", label="error")
ax.set_xlabel("width")
ax.set_ylabel("Error")
ax.legend()
plt.grid()
plt.title("Erreur de prédiction en fonction de width")