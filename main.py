# 1. Importul bibliotecilor esențiale
import pandas as pd

# 2. Încărcarea datasetului complet (cel cu valori lipsă sau fără, după cum alegi tu)
df = pd.read_csv("basketball_players_overall.csv")  # sau "basketball_players_full_over60.csv"

# 3. Afișarea primelor 5 linii pentru inspecție
print(df.head())