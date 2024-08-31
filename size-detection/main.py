from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('part_size_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to match model input size
        image = image.resize((224, 224))
        
        # Convert image to numpy array and normalize
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Predict measurements
        predictions = model.predict(image)
        measurements = predictions[0].tolist()
        
        # Return the predictions as a JSON response
        return jsonify({'measurements': measurements})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
