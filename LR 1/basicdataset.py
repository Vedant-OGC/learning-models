import pandas as pd

# Dataset
data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [30, 35, 40, 45, 50, 60, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

# features and target
X = df[["experience"]]
y = df["salary"]

print("X (features):")
print(X)

print("\ny (target):")
print(y)
