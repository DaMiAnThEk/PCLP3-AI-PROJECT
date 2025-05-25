import pandas as pd
from sklearn.model_selection import train_test_split

# Încărcarea datasetului complet
df = pd.read_csv("basketball_players_overall.csv")  # sau "basketball_players_full_over60.csv"

# Afișarea primelor 5 linii pentru inspecție
print(df.head())

# Afișare câte valori lipsă sunt pe coloană
print("Valori lipsă înainte de curățare dataset:")
print(df.isnull().sum())

# Ștergere rânduri care conțin cel puțin o valoare lipsă
df.dropna(inplace=True)

print("\nValori lipsă după curățare:")
print(df.isnull().sum())

# Definim X (caracteristici) și y (ținta de prezis - overall_rating)
X = df.drop(columns=['overall_rating'])  # totul în afară de target
y = df['overall_rating']                 # targetul

# Împărțim în train și test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Salvare set de date train și test în basketball_train.csv și basketball_test.csv:")
# Afișăm dimensiunea subseturilor
print(f"Dimensiunea setului de antrenament (X): {X_train.shape}")
print(f"Dimensiunea setului de testare (X): {X_test.shape}")

# Recompunere dataframe-urile cu X și y
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# Salvare în fișiere CSV
train_set.to_csv("basketball_train.csv", index=False)
test_set.to_csv("basketball_test.csv", index=False)
print("DONE!")

