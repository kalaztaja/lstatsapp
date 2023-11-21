from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('teampredictmodel.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        input_data = np.array(data['input'])

        # Make predictions using the loaded model
        predictions = model.predict(input_data)

        # Return the predictions as JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
