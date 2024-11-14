from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and prepare the dataset
file_path = 'exercise_dataset.csv'  # Update path if needed
fitness_data = pd.read_csv(file_path)

# Encode categorical features (Gender and Weather Conditions)
fitness_data_encoded = fitness_data.copy()
label_encoders = {}

# Encode "Gender" and "Weather Conditions" columns
for column in ['Gender', 'Weather Conditions']:
    le = LabelEncoder()
    fitness_data_encoded[column] = le.fit_transform(fitness_data[column])
    label_encoders[column] = le

# Select features and target for "Calories Burn" prediction
features = fitness_data_encoded.drop(columns=['ID', 'Exercise', 'Calories Burn'])
target = fitness_data_encoded['Calories Burn']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Linear Regression model
calories_burn_model = LinearRegression()
calories_burn_model.fit(X_train_scaled, y_train)

# Function to predict "Calories Burn" based on user input
def predict_calories_burn(age, gender, duration, heart_rate, bmi, dream_weight, actual_weight, weather_conditions, exercise_intensity):
    # Encode gender and weather conditions
    gender_encoded = label_encoders['Gender'].transform([gender])[0]
    weather_encoded = label_encoders['Weather Conditions'].transform([weather_conditions])[0]
    
    # Prepare input features
    input_data = [[age, gender_encoded, duration, heart_rate, bmi, dream_weight, actual_weight, weather_encoded, exercise_intensity]]
    
    # Scale the input features based on training data
    input_scaled = scaler.transform(input_data)
    
    # Predict and return result
    calories_burned = calories_burn_model.predict(input_scaled)[0]
    return calories_burned

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the form
        age = int(request.form['age'])
        gender = request.form['gender']
        duration = int(request.form['duration'])
        heart_rate = int(request.form['heart_rate'])
        bmi = float(request.form['bmi'])
        dream_weight = float(request.form['dream_weight'])
        actual_weight = float(request.form['actual_weight'])
        weather_conditions = request.form['weather_conditions']
        exercise_intensity = int(request.form['exercise_intensity'])

        # Predict the calories burned
        predicted_calories = predict_calories_burn(
            age=age, 
            gender=gender, 
            duration=duration, 
            heart_rate=heart_rate, 
            bmi=bmi, 
            dream_weight=dream_weight, 
            actual_weight=actual_weight, 
            weather_conditions=weather_conditions, 
            exercise_intensity=exercise_intensity
        )

        return render_template("index.html", predicted_calories=predicted_calories)

    return render_template("index.html", predicted_calories=None)

if __name__ == "__main__":
    app.run(debug=True)
