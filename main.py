import pandas as pd

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

