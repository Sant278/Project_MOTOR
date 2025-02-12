from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user input
        features = [float(request.form[f'feature{i}']) for i in range(1, 13)]
        
        # Convert features into a 2D numpy array
        features = np.array(features).reshape(1, -1)
        
        # Scale the input features
        scaled_features = scaler.transform(features)
        
        # Make a prediction
        prediction = model.predict(scaled_features)
        
        # Return the prediction result
        return render_template('index.html', prediction_text=f"Predicted Motor Speed: {prediction[0]:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
