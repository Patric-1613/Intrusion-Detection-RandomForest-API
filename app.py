from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model (make sure the model is saved as 'random_forest_model.pkl')
model = pickle.load(open('random_forest_model_10_final.pkl', 'rb'))

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle form submissions and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data from HTML
        features = [
            float(request.form['same_srv_rate']),         # 0
            float(request.form['count']),                  # 1
            float(request.form['src_bytes']),              # 2
            float(request.form['diff_srv_rate']),         # 3
            float(request.form['dst_host_srv_count']),     # 4
            float(request.form['service_private']),        # 5
            float(request.form['dst_host_same_srv_rate']), # 6
            float(request.form['serror_rate']),            # 7
            float(request.form['dst_bytes']),              # 8
            float(request.form['dst_host_diff_srv_rate'])  # 9
        ]
        
        # Convert the form data into an array and reshape it for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Use the Random Forest model to make a prediction
        prediction = model.predict(features_array)  # Ensure the model is a scikit-learn model
        
        # Prepare a message based on the prediction result
        if prediction[0] == 1:
            result = "Attack Detected"
        else:
            result = "Normal Connection"

        # Return the result to the user
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
