# Moving average Totem

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# input and clean the data
data = pd.read_csv("csv/Donnees_Comptages_Velos_Totem_Albert_1er_verbose.csv")
data = data[data.columns[0:4]]
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
data[data.columns[2]] = pd.to_numeric(
    data[data.columns[2]].str.replace("\u202f", ""))
data[data.columns[3]] = pd.to_numeric(
    data[data.columns[3]].str.replace("\u202f", ""))
data_sort = data.sort_values(by=["Date", "Heure / Time"])
data_cleaned = data_sort.drop_duplicates()

data_daily_max = data_cleaned.groupby(["Date"]).max()
data_daily_min = data_cleaned.groupby(["Date"]).min()

# Calculate the real daily nomber of bicycles
data_total_real = []
for i in range(data_daily_min.shape[0]-1):
    data_total_real.append(data_daily_min.iloc[i+1]["Vélos depuis la mise en service / Grand total"]
                           - data_daily_min.iloc[i+1]["Vélos ce jour / Today's total"] -
                           data_daily_max.iloc[i]["Vélos depuis la mise en service / Grand total"] +
                           data_daily_max.iloc[i]["Vélos ce jour / Today's total"])
data_total_real.append(
    data_daily_max.iloc[-1]["Vélos ce jour / Today's total"])

data_daily_total = pd.DataFrame(
    data_daily_max["Vélos ce jour / Today's total"])
data_daily_total["real_total"] = data_total_real

# Add day of week
weekDays = ("Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday")
weekdays_list = list(data_daily_total.index.weekday)
weekdays = []
for i in weekdays_list:
    weekdays.append(weekDays[i])
data_daily_total["weekday"] = weekdays

# export csv
# data_daily_total.to_csv("Donnees_Comptages_Velos_Totem_Albert_1er_verbose_par_jour_real.csv")


# Plot data
fig = plt.figure(figsize=(16, 9))
ax = plt.axes()
ax.plot(data_daily_total["Vélos ce jour / Today's total"],
        "g.-", label="input")
ax.plot(data_daily_total["real_total"], "r,-", label="output")
ax.set_xlabel("Date")
ax.set_ylabel("Daily total numbers of bicycles")
ax.legend()
plt.grid()


# Moving average
width = 7
raw_x = np.array(data_daily_total["Vélos ce jour / Today's total"])
raw_y = np.array(data_daily_total["real_total"])
T = len(raw_x)
# Build the matrix A
A = np.zeros((T, width))
A[0:, 0] = raw_x[0:]
for i in range(1, width):
    A[i:, i] = raw_x[0:-i]
wop = np.dot(np.linalg.pinv(A), raw_y)
y_pred = np.dot(A, wop)
data_daily_total["prediction"] = y_pred
# Plot
fig = plt.figure(figsize=(16, 9))
ax = plt.axes()
ax.plot(data_daily_total["real_total"], "r.-", label="real_output")
ax.plot(data_daily_total["prediction"], "b.-", label="prediction")
#ax.plot(data_daily_total["Vélos ce jour / Today's total"],"g.-",label="input")
ax.set_xlabel("Date")
ax.set_ylabel("Daily total numbers of bicycles")
ax.legend()
plt.grid()


# Calculate the error
err = []
max_window = 180
raw_x = np.array(data_daily_total["Vélos ce jour / Today's total"])
raw_y = np.array(data_daily_total["real_total"])
T = len(raw_x)
for width in range(1, max_window+1):
    A = np.zeros((T, width))
    A[0:, 0] = raw_x[0:]
    for i in range(1, width):
        A[i:, i] = raw_x[0:-i]
    wop = np.dot(np.linalg.pinv(A), raw_y)
    y_pred = np.dot(A, wop)
    err.append(np.linalg.norm(y_pred-raw_y))
# Plot
fig = plt.figure(figsize=(16, 9))
ax = plt.axes()
ax.plot(err, ".-", label="error")
ax.set_xlabel("width")
ax.set_ylabel("Error")
ax.legend()
plt.grid()