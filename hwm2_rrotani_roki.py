import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
#per il punto2:
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
#per il punto3:
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# per il punto4:
from sklearn.neighbors import KNeighborsClassifier
# per il punto4.2 -> Naive BBayes:
from sklearn.naive_bayes import GaussianNB
# per il punto 5:
import seaborn as sns

# Dopo aver importato le librerie (qua sopra indicate: )
# Devo importare il dataset , ovver il file fornito dalla pagina del homework2

from google.colab import files
uploaded = files.upload()

# Una volta scelto il file, dalla mia cartella homework2, preso dal sito di DB
# DEvo leggerlo il dataset:
df = pd.read_excel('dataset_breast.xlsx', engine='openpyxl')  # Legge il file Excel


#ora passiamo all'esplorazione del dataset:
# visualizza le prime righe del dataset
df.head()
#informazioni sul dataset
df.info()
#Statistiche
df.describe()

# Ok, tutto questo è servito per esplorare il dataset.
# ora passo a idenfiticare i problemi:
# 1- cerco valori mancanti, 2- controllo gli errori nei dati
# 3- idenfitico la class e le colonne predittive
df.isnull().sum()

# Visualizzare le prime righe del dataset:
print("Prime righe del dataset: ")
print(df.head())
# df.head mostra le prime righe del dataset e serve a capire
# comè struttorato il dataset e verifico anche che sia stato caricato
# in maniera corretta.

# Informazioni generali sul dataset:
print("\nInformazioni generali: ")
print(df.info())
# df.info fornisce informazioni del tipo, ntotale righe e colonne,
# nomi colonne, tipo di dati e numero di valori nulll.

# Statistiche descrittive:
print("\nStatistiche descrittive: ")
print(df.describe())
# df.describe calcola le stat di base per le colonne numeriche, media-max....
# serve per individuare valori insoliti outlier

# Controllo dei valori mancanti
print("\nValori mancanti per colonna: ")
print(df.isnull().sum())
# df.isnull crea una tabella con valori booleani dove T indica il valore mancante
# .sum calcola il tot dei valori mancanti MAAAA per ogni colonna.
# cosi capisco se devo gestire qualche dato mancante!

# Separiamo la variabile target (y) e le variabili predittive (X)
X = df.drop('Class', axis=1)  # Tutte le colonne tranne 'Class'
y = df['Class']  # La colonna target
print("Variabili indipendenti (X):")
print(X.head())
print("\nVariabile target (y):")
print(y.head())

# Applicazione della codifica One-Hot alle variabili categoriche
X_encoded = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding
print("Variabili dopo la codifica One-Hot:")
print(X_encoded.head())
# pd.get_dummies() converte le variabili categoriche in mumeriche
# drop_first=True elimina una delle cat pr evitare la multicolinearità

# Mostrare quante colonne e righe ci sono nelle variabili
print("Forma delle variabili predittive dopo la codifica:", X_encoded.shape)
print("Numero totale di campioni:", len(y))

# 1 PUNTO 1 - - - - - - - -
print("\nPUNTO 1: ------------")
# Creazione del modello di albero decisionale
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_encoded, y)
# Predizioni
y_pred = dt_model.predict(X_encoded)
# Valutazione
print("Accuracy dell'albero decisionale:", accuracy_score(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))
# Visualizzazione dell'albero
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X_encoded.columns, class_names=dt_model.classes_, filled=True)
plt.show()

# per rispondere ai punti 1.a e 1.b aggiungo questi:
# Calcolo dell'importanza delle feature
feature_importances = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Attributo più discriminante:")
print(feature_importances.head(1))  # Mostra il più importante
#1b:
print("Altezza dell'albero decisionale:", dt_model.get_depth())

# PUNTO 2:
print("\nPUNTO 2 --------------------------")
# Configurazioni dei parametri
param_grid = {
    'min_impurity_decrease': [0.0, 0.01, 0.02],
    'min_samples_leaf': [1, 5, 10],
    'max_depth': [3, 5, 7]
}

# Iterare sulle configurazioni
tree_count = 0
for mid in param_grid['min_impurity_decrease']:
    for msl in param_grid['min_samples_leaf']:
        for md in param_grid['max_depth']:
            tree_count += 1
            if tree_count > 5:  # Limitiamoci a 5 alberi per ora
                break
            # Creazione del modello
            dt_model = DecisionTreeClassifier(
                criterion="entropy",  # Suddivisione basata sull'entropia
                min_impurity_decrease=mid,
                min_samples_leaf=msl,
                max_depth=md,
                random_state=42
            )
            dt_model.fit(X_encoded, y)
            # Predizioni
            y_pred = dt_model.predict(X_encoded)
            accuracy = accuracy_score(y, y_pred)
            # Visualizzazione del modello
            print(f"Configurazione {tree_count}: min_impurity_decrease={mid}, min_samples_leaf={msl}, max_depth={md}")
            print(f"Accuracy: {accuracy:.4f}")
            plt.figure(figsize=(20, 10))
            plot_tree(dt_model, feature_names=X_encoded.columns, class_names=dt_model.classes_, filled=True)
            plt.title(f"Decision Tree {tree_count}")
            plt.show()

# PUNTO 3: 10- FOLD. CROSS-Validation:
print("\nPUNTO 3 --------------------------")
# start:
# Configurazioni dei parametri
param_grid = {
    'min_impurity_decrease': [0.0, 0.01],
    'min_samples_leaf': [1, 5],
    'max_depth': [3, 5]
}

# Cross-validation stratificata
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Loop sulle configurazioni
config_count = 0
for mid in param_grid['min_impurity_decrease']:
    for msl in param_grid['min_samples_leaf']:
        for md in param_grid['max_depth']:
            config_count += 1
            if config_count > 5:  # Limitiamoci a 5 configurazioni per ora
                break
            # Creazione del modello
            dt_model = DecisionTreeClassifier(
                criterion="entropy",
                min_impurity_decrease=mid,
                min_samples_leaf=msl,
                max_depth=md,
                random_state=42
            )
            # Accuratezza media con cross-validation
            accuracies = []
            y_true_all, y_pred_all = [], []
            for train_index, test_index in skf.split(X_encoded, y):
                X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                dt_model.fit(X_train, y_train)
                y_pred = dt_model.predict(X_test)

                accuracies.append(accuracy_score(y_test, y_pred))
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)
            # Accuratezza media
            accuracy_mean = sum(accuracies) / len(accuracies)
            print(f"Configurazione {config_count}: min_impurity_decrease={mid}, min_samples_leaf={msl}, max_depth={md}")
            print(f"Accuratezza media (10-fold CV): {accuracy_mean:.4f}")
            # Matrice di confusione
            cm = confusion_matrix(y_true_all, y_pred_all, labels=dt_model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
            disp.plot(cmap="Blues")
            plt.title(f"Confusion Matrix - Configurazione {config_count}")
            plt.show()

# PUNTO 4:
print("\nPUNTO 4 --------------------------")
#start:
# Configurazioni di K
k_values = [1, 3, 5, 7, 9]
# Loop su diversi valori di K
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)

    # Cross-validation stratificata
    accuracies = []
    y_true_all, y_pred_all = [], []
    for train_index, test_index in skf.split(X_encoded, y):
        X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    # Accuratezza media
    accuracy_mean = sum(accuracies) / len(accuracies)
    print(f"K-NN con K={k}: Accuratezza media (10-fold CV): {accuracy_mean:.4f}")
    # Matrice di confusione
    cm = confusion_matrix(y_true_all, y_pred_all, labels=knn_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_model.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - K={k}")
    plt.show()

# PUNTO 4.2:
print("\n Punto 4.2 ----------------")
#start:
# Modello Naïve Bayes
nb_model = GaussianNB()
# Cross-validation stratificata
accuracies = []
y_true_all, y_pred_all = [], []
for train_index, test_index in skf.split(X_encoded, y):
    X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
# Accuratezza media
accuracy_mean = sum(accuracies) / len(accuracies)
print(f"Naïve Bayes: Accuratezza media (10-fold CV): {accuracy_mean:.4f}")
# Matrice di confusione
cm = confusion_matrix(y_true_all, y_pred_all, labels=nb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Naïve Bayes")
plt.show()

# PUNTO 5: FINALE:
print("\nPUNTO 5 --------------------------")
#start
# Calcolo della matrice di correlazione
correlation_matrix = X_encoded.corr()
# Visualizzazione della matrice di correlazione
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matrice di correlazione")
plt.show()
# Coppia di attributi più correlata
correlation_pairs = correlation_matrix.unstack().sort_values(ascending=False)
most_correlated = correlation_pairs[correlation_pairs < 1].idxmax()  # Escludiamo la diagonale
highest_correlation = correlation_pairs[most_correlated]
print(f"Coppia di attributi più correlata: {most_correlated} con correlazione = {highest_correlation:.2f}")