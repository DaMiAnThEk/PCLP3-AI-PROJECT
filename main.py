import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# Eliminăm 'name' și aplicăm One-Hot Encoding
df = df.drop(columns=['name'])
df = pd.get_dummies(df, columns=['position', 'team'], drop_first=True)

X = df.drop(columns=['overall_rating'])
y = df['overall_rating']

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

# Analiza pe dataset
numeric_df = df.select_dtypes(include=['number'])
# Matricea de corelație
corr_matrix = numeric_df.corr()
# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matricea de corelație între variabile numerice")
plt.tight_layout()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matricea de corelație între variabile numerice")
plt.tight_layout()
plt.savefig("heatmap_corelatii.png")

#Antrenarea și evaluarea modelului
from sklearn.linear_model import LinearRegression

# Creare model
model = LinearRegression()

# Antrenare model pe setul de antrenament
model.fit(X_train, y_train)

# Prezicere overall pentru datele de test
y_pred = model.predict(X_test)

# Primele 5 predicții
print("Predicții:", y_pred[:5])
print("Adevărate:", y_test[:5].values)

# Evaluarea performanței
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")