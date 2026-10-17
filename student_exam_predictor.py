import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

print("====================================")
print(" STUDENT EXAM PERFORMANCE PREDICTOR")
print("====================================")
print("Using Multiple Linear Regression")
print()

data = pd.read_csv("student_data.csv")

print(data)

X = data[["StudyHours", "PreviousScore", "Attendance"]]

y = data["FinalScore"]

print("X:")
print(X)

print("y:")
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:")
print(X_train)

print("Testing data:")
print(X_test)


model = LinearRegression()

model.fit(X_train, y_train)


study_hours = float(input("Enter study hours: "))
previous_score = float(input("Enter previous score: "))
attendance = float(input("Enter attendance percentage: "))

new_student = pd.DataFrame(
    [[study_hours, previous_score, attendance]],
    columns=["StudyHours", "PreviousScore", "Attendance"]
)

prediction = model.predict(new_student)[0]

print("\n----- STUDENT EXAM RESULT -----")
print("Predicted Final Score:", round(prediction, 2))

if prediction >= 40:
    print("Predicted Result: PASS")
else:
    print("Predicted Result: FAIL")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 Score:", r2)
print("Mean Squared Error:", mse)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)