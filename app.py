
from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

app = Flask(__name__)

# Load the dataset
file_path = Path(__file__).with_name("student_data.csv")
data = pd.read_csv(file_path)

# Prepare the data
X = data[["StudyHours", "PreviousScore", "Attendance"]]
y = data["FinalScore"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Website design
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Exam Performance Predictor</title>
    <style>
        body {
            font-family: Arial;
            background: #eaf2f8;
            text-align: center;
            padding: 30px;
        }
        .box {
            background: white;
            padding: 25px;
            max-width: 400px;
            margin: auto;
            border-radius: 12px;
        }
        input {
            width: 90%;
            padding: 10px;
            margin: 8px;
        }
        button {
            padding: 12px 25px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 5px;
        }
        h1 { color: #1d4ed8; }
    </style>
</head>
<body>
    <div class="box">
        <h1>Student Exam Predictor</h1>
        <p>Enter the student's details to predict the final score.</p>

        <form method="POST">
            <input type="number" step="any" name="hours"
                   placeholder="Study Hours" min="0" max="24" required>

            <input type="number" step="any" name="previous"
                   placeholder="Previous Score (0-100)"
                   min="0" max="100" required>

            <input type="number" step="any" name="attendance"
                   placeholder="Attendance (0-100%)"
                   min="0" max="100" required>

            <button type="submit">Predict Score</button>
        </form>

        {% if prediction is not none %}
            <h2>Predicted Final Score: {{ prediction }}/100</h2>
            <h3>Predicted Result: {{ result }}</h3>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    result = ""

    if request.method == "POST":
        hours = float(request.form["hours"])
        previous = float(request.form["previous"])
        attendance = float(request.form["attendance"])

        student = pd.DataFrame(
            [[hours, previous, attendance]],
            columns=["StudyHours", "PreviousScore", "Attendance"]
        )

        prediction = round(float(model.predict(student)[0]), 2)
        prediction = max(0, min(100, prediction))

        if prediction >= 40:
            result = "PASS"
        else:
            result = "FAIL"

    return render_template_string(
        html, prediction=prediction, result=result
    )

if __name__ == "__main__":
    app.run(debug=True)
