from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Charger les données
data = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)