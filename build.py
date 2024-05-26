import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Suppress FutureWarning related to Downcasting behavior
pd.set_option('mode.chained_assignment', None)

# Load data
train_df = pd.read_json('data/train.json')
test_df = pd.read_json('data/test.json')

# Handle missing values in train_df
train_df['inc_angle'] = train_df['inc_angle'].replace('na', np.nan).astype(float)
train_df['inc_angle'].fillna(train_df['inc_angle'].mean(), inplace=True)

# Preprocess image data
X_train = np.array([np.array(band).reshape(75, 75) for band in train_df['band_1']])
X_test = np.array([np.array(band).reshape(75, 75) for band in test_df['band_1']])
X_train = X_train[:, :, :, np.newaxis]
X_test = X_test[:, :, :, np.newaxis]

y_train = train_df['is_iceberg']

# Normalize image data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(75, 75, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_train, y_train)
print(f'Training Accuracy: {accuracy:.2f}')

# Save model to HDF5 file
model.save('iceberg_detection_model.h5')

# Make predictions on test data
predictions = model.predict(X_test)

# Prepare submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'is_iceberg': predictions.reshape(-1)
})

submission_df.to_csv('submission.csv', index=False)
