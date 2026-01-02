import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [30, 35, 40, 45, 50, 60, 65, 70, 80, 90]
}

df = pd.read_csv(r"C:\Users\NEWTON\Desktop\Models\LR 1\Salary_dataset.csv")
df.head()

X = df[["YearsExperience"]]
y = df["Salary"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("X_test (experience):")
print(X_test.values.flatten())

print("\nActual salaries:")
print(y_test.values)

print("\nPredicted salaries:")
print(y_pred)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)

