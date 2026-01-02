import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [30, 35, 40, 45, 50, 60, 65, 70, 80, 90]
}

df = pd.DataFrame(data)


X = df[["experience"]]
y = df["salary"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully")

# Show learned parameters
print("Slope (coefficient):", model.coef_)
print("Intercept:", model.intercept_)
