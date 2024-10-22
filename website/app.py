from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import io
import base64
import logging
import numpy as np
from bson import ObjectId
import os
os.environ['TEAM_API_KEY'] = '64c4a9fb8d26148cdf7ca52fbd515108caa5bbe3ff9827babdb150801c561a3f'

import secrets

from aixplain.factories import ModelFactory

app = Flask(__name__)

# Initialize the aiXplain model
model = ModelFactory.get("6414bd3cd09663e9225130e8")  # Replace with your aiXplain model ID



app = Flask(__name__)

app.config["MONGO_URI"] = "mongodb://aqsashah:mariyak439#@localhost:27017/h20manager"

app.config['SECRET_KEY'] = "h!3K$s1#m2@aF1%PnJ8&X#4zL@qR!"
  # Generates a random 24-byte key

mongo = PyMongo(app)

# Route for the landing page (index)
@app.route('/')
def index():
    return render_template('index.html')

def serialize_object(obj):
    """Helper function to serialize MongoDB's ObjectId to string."""
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError("Type not serializable")

# Function to load the dataset and preprocess it
def load_and_preprocess_data():
    df = pd.read_csv(r'C:\Users\AQSA SHAH\OneDrive\Desktop\h20manager(deployed)\website\DATASET - Sheet1.csv')
    df['WATER REQUIREMENT'] = pd.to_numeric(df['WATER REQUIREMENT'], errors='coerce')
    df.dropna(subset=['WATER REQUIREMENT'], inplace=True)

    categorical_columns = ['CROP TYPE', 'SOIL TYPE', 'REGION', 'TEMPERATURE', 'WEATHER CONDITION']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    X = df_encoded.drop('WATER REQUIREMENT', axis=1)
    y = df_encoded['WATER REQUIREMENT']

    return X, y, df_encoded

# Function to train and save the model
def train_model():
    X, y, _ = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, r'C:\Users\AQSA SHAH\OneDrive\Desktop\h20manager(deployed)\website\water_model.pkl')

# Function to load the trained model
def load_model():
    return joblib.load(r'C:\Users\AQSA SHAH\OneDrive\Desktop\h20manager(deployed)\website\water_model.pkl')



# Route for the analytics page
@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('analytics.html')

# Route for the forecast page with water requirement prediction
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated

    if request.method == 'POST':
        crop = request.form['crop']
        soil = request.form['soil']
        region = request.form['region']
        temperature = request.form['temperature']
        weather = request.form['weather']

        _, _, df_encoded = load_and_preprocess_data()

        input_data = {
            'CROP TYPE_' + crop: 1,
            'SOIL TYPE_' + soil: 1,
            'REGION_' + region: 1,
            'TEMPERATURE_' + temperature: 1,
            'WEATHER CONDITION_' + weather: 1
        }

        input_df = pd.DataFrame(0, index=[0], columns=df_encoded.columns.drop('WATER REQUIREMENT'))
        input_df.update(pd.DataFrame(input_data, index=[0]))

        model = load_model()
        prediction = model.predict(input_df)[0]

        return render_template('forecast.html', prediction=prediction)

    return render_template('forecast.html')

# Route for the reservoirs page
@app.route('/reservoirs')
def reservoirs():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('reservoirs.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Extract user data from form
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if user already exists
        existing_user = mongo.db.users.find_one({'email': email})
        if existing_user:
            return "User already exists!"

        # Hash the password before storing it
        hashed_password = generate_password_hash(password)

        # Insert user data into MongoDB
        mongo.db.users.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password
        })

        return redirect(url_for('login'))  # Redirect to login page after successful signup

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Find the user by email
        user = mongo.db.users.find_one({'email': email})

        if user and check_password_hash(user['password'], password):
            # Create session for the logged-in user
            session['user_id'] = str(user['_id'])
            print(f"Logged in user ID: {session['user_id']}")  # Check if session is set
            return redirect(url_for('index'))  # Redirect to the home page after successful login
        else:
            # If credentials are invalid, pass an error message
            error = "Invalid email or password. Please try again."
            return render_template('login.html', error=error)
        
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Clear the session
    return redirect(url_for('index'))  


@app.route('/chatbot')
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('chatbot.html')

# Route for the footprint page
@app.route('/footprint')
def footprint():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('footprint.html')

@app.route('/analyze-water-usage', methods=['POST'])
def analyze_water_usage():
    data = request.get_json()

    # Ensure all required months are present
    if not all(month in data for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec']):
        return jsonify({'success': False, 'error': 'Missing data for one or more months'})

    try:
        # Convert the values to float for plotting
        usage_data = [float(data[month]) for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec']]
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid input, all fields must be numeric'})

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    
    # Generate the plot
    plt.figure(figsize=(10, 5))
    plt.plot(months, usage_data, marker='o', color='cyan', label='Water Usage (liters)')
    plt.xlabel('Months', color='white')
    plt.ylabel('Water Usage (liters)', color='white')
    plt.title('Monthly Water Usage Analysis', color='white')
    plt.grid(True)
    plt.legend()

    # Set plot background to black
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    plt.tick_params(colors='white')

    # Save plot as image and encode to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor='black')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({'success': True, 'graph': graph_url})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    text = f"Water usage analysis:\nJanuary: {data['january']} liters\nFebruary: {data['february']} liters\nMarch: {data['march']} liters\nApril: {data['april']} liters\nMay: {data['may']} liters\nJune: {data['june']} liters\nJuly: {data['july']} liters\nAugust: {data['august']} liters\nSeptember: {data['september']} liters\nOctober: {data['october']} liters\nNovember: {data['november']} liters\nDecember: {data['december']} liters"
    
    result = model.run({'text': text})

    if result.get('completed'):
        return jsonify({'result': result['data']})
    else:
        return jsonify({'result': 'No detailed report generated. Please check the AI model.'})

@app.route('/report')
def report():
    return render_template('report.html')  # Ensure report.html is in the templates folder





if __name__ == "__main__":
    app.run(debug=True)
