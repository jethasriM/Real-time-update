import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# Generate synthetic training and test data
X_train = np.random.random((100, 224, 224, 3))
y_train = np.random.random((100, 4))
X_test = np.random.random((20, 224, 224, 3))
y_test = np.random.random((20, 4))

# Normalize the data (if your data was in the range [0, 255])
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# Load the base model with pre-trained weights from ImageNet, without the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Freeze the layers of the base model to prevent them from being updated during training
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='linear')(x)

# Create the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[early_stopping])

# Save the trained model
model.save('part_size_model.h5')
