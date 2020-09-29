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

# L'input est composé de toutes les données :
input = accidents_week
# L'output est composé des données de 2017 uniquement :
output = accidents_week.iloc[58561:63389:, :]
accidents_week = input.groupby(['weekday', input.index.hour])[
    'sexe'].count().unstack(level=0)
accidents_d = accidents_week.sum()
# Nombre d'accidents par jour de la semaine en moyenne par année :
print(accidents_d/14)
accidents_d = accidents_d/14
accidents_week_2017 = output.groupby(['weekday', output.index.hour])[
    'sexe'].count().unstack(level=0)
accidents_d_2017 = accidents_week_2017.sum()
# Nombre d'accidents par jour de la semaine en 2017 :
print(accidents_d_2017)
accident_2017 = accidents_d_2017

np.set_printoptions(threshold=sys.maxsize)
width = 6
raw_x = np.array(accidents_d)
raw_y = np.array(accident_2017)
# Plot de l'input et de l'output
fig = plt.figure(figsize=(7, 7))
ax = plt.axes()
ax.plot(raw_x, "r.-", label="input")
ax.plot(raw_y, "b.-", label="output")
ax.set_xlabel("Jour de la semaine")
ax.set_ylabel("Accidents")
ax.legend()
plt.title("Moyenne du nombe d'accidents par jour de la semaine (Lundi = 0, ..., Dimanche=6)")
plt.grid()

T = len(raw_x)
A = np.zeros((T, width))
A[0:, 0] = raw_x[0:]
for i in range(1, width):
    # On crée la matrice qui va permettre de résoudre l'équation Y=AW, avec W la matrice de coefficients du modèle
    A[i:, i] = raw_x[0:-i]

# Résultat de A^-1Y=W
wop = np.dot(np.linalg.pinv(A), raw_y)
# Grâce aux coefficients estimés, on peut estimer Y :
y_pred = np.dot(A, wop)

# Comparaison de la prédiction et de la vraie valeur de Y : avec width = 6, on obtient deux courbes quasi superposées.
fig = plt.figure(figsize=(7, 7))
ax = plt.axes()
ax.plot(accident_2017, "r.-", label="real_output")
ax.plot(y_pred, "b.-", label="prediction")
ax.set_xlabel("Jour de la semaine")
ax.set_ylabel("Accidents")
ax.legend()
plt.grid()
plt.title("Moyenne du nombe d'accidents par jour de la semaine en 2017 (Lundi = 0, ..., Dimanche=6)")
# Les deux courbes sont presques superposées, la prédiction semble très bonne. On remarque que c'est le mercredi qui compte le plus d'accidents, peut-être car c'est un jour où les plus jeunes n'ont pas cours et sortent en vélo ? Cependant, le dimanche compte très peu d'accident, donc cette théorie semble ne pas tenir la route.

# Calcul de l'erreur pour différentes valeurs de witdh.
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

# Plot de l'erreur : dès width >= 6, l'erreur est faible, la prédiction sera donc très proche de la vraie valeur pour width>= 6.
fig = plt.figure(figsize=(16, 9))
ax = plt.axes()
ax.plot(err, ".-", label="error")
ax.set_xlabel("width")
ax.set_ylabel("Error")
ax.legend()
plt.grid()
plt.title("Erreur de prédiction en fonction de width")