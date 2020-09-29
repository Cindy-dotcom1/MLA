import pandas as pd
from download import download
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact
import sys

pd.set_option('display.max_rows', 50000)

df_bikes = pd.read_csv("bicycle_db.csv",
                       na_values="", converters={'data': str, 'heure': str})
df_bikes['heure'] = df_bikes['heure'].replace('', np.nan)
df_bikes.iloc[400:402]
df_bikes.dropna(subset=['heure'], inplace=True)
df_bikes.iloc[399:402]
df_bikes['date'] + ' ' + df_bikes['heure'] + ':00'
# on change le format pour la date et l'horaire pour qu'il corresponde au format usuel :

time_improved = pd.to_datetime(df_bikes['date'] +
                               ' ' + df_bikes['heure'] + ':00',
                               format='%Y-%m-%d %H:%M')

# avec d = day, m=month, Y=year, H=hour, M=minutes
df_bikes['Time'] = time_improved
df_bikes.set_index('Time', inplace=True)
# on enlève les colonnes inutiles
del df_bikes['heure']
del df_bikes['date']

# Chargement des couleurs
sns.set_palette("colorblind", n_colors=7)
df_bikes['weekday'] = df_bikes.index.weekday  # Monday=0, Sunday=6

days = ['Lundi', 'Mardi', 'Mercredi',
        'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

accidents_week = df_bikes.groupby(['weekday', df_bikes.index.hour])[
    'sexe'].count().unstack(level=0)


fig, axes = plt.subplots(1, 1, figsize=(7, 7))


accidents_week.plot(ax=axes)
axes.set_ylabel("Accidents")
axes.set_xlabel("Heure de la journée")
axes.set_title(
    "Profil journalier des accidents: effet du weekend?")
axes.set_xticks(np.arange(0, 24))
axes.set_xticklabels(np.arange(0, 24), rotation=45)
axes.legend(labels=days, loc='lower left', bbox_to_anchor=(1, 0.1))

plt.tight_layout()

accidents_week = df_bikes.sort_values(['Time'])
# L'input est composé de toutes les données :
input = accidents_week
# L'output est composé des données de 2018 uniquement :
output = accidents_week.iloc[63389:, :]

accidents_week = input.groupby(['weekday', input.index.hour])[
    'sexe'].count().unstack(level=0)
accidents_d = accidents_week.sum()
# Nombre d'accidents par jour de la semaine en moyenne par année :
print(accidents_d/14)
accidents_d = accidents_d/14
accidents_week_2018 = output.groupby(['weekday', output.index.hour])[
    'sexe'].count().unstack(level=0)
accidents_d_2018 = accidents_week_2018.sum()
# Nombre d'accidents par jour de la semaine en 2018 :
print(accidents_d_2018)
accident_2018 = accidents_d_2018

np.set_printoptions(threshold=sys.maxsize)

width = 6
raw_x = np.array(accidents_d)
raw_y = np.array(accident_2018)
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
y_pred = np.dot(A, wop)

# Comparaison de la prédiction et de la vraie valeur de Y : avec width = 6, on obtient deux courbes quasi superposées. On remarque que comme pour 2017, le mercredi est un jour avec beaucoup d'accidents contrairement au week-end
fig = plt.figure(figsize=(7, 7))
ax = plt.axes()
ax.plot(accident_2018, "r.-", label="real_output")
ax.plot(y_pred, "b.-", label="prediction")
ax.set_xlabel("Jour de la semaine")
ax.set_ylabel("Accidents")
ax.legend()
plt.grid()
plt.title("Moyenne du nombe d'accidents par jour de la semaine en 2018 (Lundi = 0, ..., Dimanche=6)")

# Quelle valeur aurait-il fallu choisir pour width ? Pour le savoir, on va calculer l'erreur de prédiction pour plusieurs valeurs de width.
err = []
max_window = 200
raw_x = np.array(np.array(accidents_d))
raw_y = np.array(np.array(accident_2018))
T = len(raw_x)
for width in range(1, max_window+1):
    A = np.zeros((T, width))
    A[0:, 0] = raw_x[0:]
    for i in range(1, width):
        A[i:, i] = raw_x[0:-i]
    wop = np.dot(np.linalg.pinv(A), raw_y)
    y_pred = np.dot(A, wop)
    err.append(np.linalg.norm(y_pred-raw_y))

# Plot de l'erreur : dès width >= 7, l'erreur tend vers 0, la prédiction sera donc très proche de la vraie valeur pour width>= 7.
fig = plt.figure(figsize=(16, 9))
ax = plt.axes()
ax.plot(err, ".-", label="error")
ax.set_xlabel("width")
ax.set_ylabel("Error")
ax.legend()
plt.grid()
plt.title("Erreur de prédiction en fonction de width")
