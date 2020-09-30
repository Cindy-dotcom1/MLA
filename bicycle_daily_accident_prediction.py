import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# input and clean the data
df_bikes = pd.read_csv("csv/accident-velo.csv", na_values="")
df_bikes["date"] = pd.to_datetime(df_bikes["date"])
df_bikes_sort = df_bikes.sort_values(by=["date"])


# Count the daily number of accidents
df_bikes_date = pd.DataFrame(df_bikes_sort.groupby(
    "date").count()["identifiant accident"])
# Group the data by year
df_bikes_day = df_bikes_date.groupby(df_bikes_date.index.year)


# Build the dataframe which contains the daily number of accidents grouped by year
df_bikes_2005 = df_bikes_day.get_group(2005)  # 2005
datamissing = pd.to_datetime(pd.date_range(start='2005-01-01',
                                           end='2005-12-31').difference(df_bikes_2005.index))  # Find the dates when there is no accident (missing dates)
# replace the number of accidents with 0 for the missing dates
data = {"identifiant accident": 0}
df_bikes_2005 = df_bikes_2005.append(pd.DataFrame(data, index=datamissing))
df_bikes_2005 = df_bikes_2005.sort_index(axis=0)
df_bikes_2005 = df_bikes_2005.rename(
    columns={"identifiant accident": "2005"})  # rename the column

# We do the same thing for the year 2006-2017, but there are leap years (2008,2012,2016),
# so in this case we delete February 29. We put all the data in the dataframe df_bikes_2005.


def create_df_day(year, leap_year, df_bikes_day, df_bikes_2005):
    df_bikes_year = df_bikes_day.get_group(year)
    year_start = str(year)+'-01-01'
    year_end = str(year)+'-12-31'
    datamissing = pd.to_datetime(pd.date_range(start=year_start,
                                               end=year_end).difference(df_bikes_year.index))
    if year in leap_year:
        extra_date = str(year)+'-02-29'
        df_bikes_year = df_bikes_year.drop(pd.to_datetime(extra_date))
    data = {"identifiant accident": 0}
    df_bikes_year = df_bikes_year.append(pd.DataFrame(data, index=datamissing))
    df_bikes_year = df_bikes_year.sort_index(axis=0)
    df_bikes_2005[year] = df_bikes_year["identifiant accident"].values
    return(df_bikes_2005)


leap_year = [2008, 2012, 2016]
for year in range(2006, 2018):
    df_bikes_2005 = create_df_day(year, leap_year, df_bikes_day, df_bikes_2005)

# Calculate the mean value of each row of the dataframe df_bikes_2005
# df_bikes_2005['mean'] will be the input
df_bikes_2005['mean'] = df_bikes_2005.mean(axis=1)


# We do the same thing for 2018 (this will be output)
df_bikes_2018 = df_bikes_day.get_group(2018)
datamissing = pd.to_datetime(pd.date_range(start='2018-01-01',
                                           end='2018-12-31').difference(df_bikes_2018.index))
data = {"identifiant accident": 0}
df_bikes_2018 = df_bikes_2018.append(pd.DataFrame(data, index=datamissing))
df_bikes_2018 = df_bikes_2018.sort_index(axis=0)


# Plot data
fig = plt.figure(figsize=(16, 9))
ax = plt.axes()
ax.plot(df_bikes_2005['mean'].values, "g.-", label="input")
ax.plot(df_bikes_2018["identifiant accident"].values, "r.-", label="output")
ax.set_xlabel("Date")
ax.set_ylabel("nombre of accidents")
ax.legend()
plt.grid()


# Moving average
width = 7
raw_x = np.array(df_bikes_2005['mean'].values)  # input
raw_y = np.array(df_bikes_2018["identifiant accident"].values)  # output
T = len(raw_x)
# Build the matrix A
A = np.zeros((T, width))
A[0:, 0] = raw_x[0:]
for i in range(1, width):
    A[i:, i] = raw_x[0:-i]
wop = np.dot(np.linalg.pinv(A), raw_y)
y_pred = np.dot(A, wop)

# Plot data
fig = plt.figure(figsize=(16, 9))
ax = plt.axes()
ax.plot(df_bikes_2018["identifiant accident"].values,
        "r.-", label="real_output")
ax.plot(y_pred, "b.-", label="prediction")
ax.set_xlabel("Date")
ax.set_ylabel("nombre of accidents")
ax.legend()
plt.grid()


# Calculate the error
err = []
max_window = 365
raw_x = np.array(df_bikes_2005['mean'].values)
raw_y = np.array(df_bikes_2018["identifiant accident"].values)
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