import pandas as pd
from sklearn.model_selection import train_test_split

# dataset
data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [30, 35, 40, 45, 50, 60, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

# features and target
X = df[["experience"]]
y = df["salary"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data (X_train):")
print(X_train)

print("\nTesting data (X_test):")
print(X_test)

print("\nTraining labels (y_train):")
print(y_train)

print("\nTesting labels (y_test):")
print(y_test)
