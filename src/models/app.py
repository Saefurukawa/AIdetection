from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your scikit-learn model using joblib
model = joblib.load('complementNB_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()
        # Assume the data is a list of inputs for prediction
        # You might need to preprocess the data before feeding it to the model
        predictions = model.predict(data)
        return jsonify(predictions.tolist())  # Return the predictions as a JSON response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Run the app on localhost, port 5000
