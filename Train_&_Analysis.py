import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Parameters
image_height = 256
image_width = 256
num_channels = 1  # Grayscale images have one channel
epochs = 50
batch_size = 32


# Load the data
def load_data(data_dir):
    images = []
    labels = []
    for grid_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, grid_dir)):
            grid_label = list(map(int, grid_dir.split('_')[1:]))
            grid_path = os.path.join(data_dir, grid_dir)
            for img_file in os.listdir(grid_path):
                img_path = os.path.join(grid_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
                if img is not None:
                    img_resized = cv2.resize(img, (image_height, image_width))
                    images.append(img_resized)
                    labels.append(grid_label)
    return np.array(images), np.array(labels)


# Assume your data is saved in 'eye_images_gray' directory
data_dir = 'eye_gray_mixed'
images, labels = load_data(data_dir)

# Normalize images
images = images / 255.0

# Add a channel dimension to the images
images = np.expand_dims(images, axis=-1)  # Now the shape is (num_samples, 256, 256, 1)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


# Define the CNN model
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2)  # Output layer with 2 neurons for x and y coordinates
    ])
    return model


input_shape = (image_height, image_width, num_channels)
model = create_model(input_shape)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())


# Define a custom callback to calculate and store metrics
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_images, val_labels, epochs_to_evaluate):
        self.val_images = val_images
        self.val_labels = val_labels
        self.epochs_to_evaluate = epochs_to_evaluate
        self.epochs = []
        self.mse = []
        self.mae = []
        self.r2 = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) in self.epochs_to_evaluate:
            # Predict on validation set
            predictions = self.model.predict(self.val_images)
            mse = np.mean(np.square(predictions - self.val_labels))
            mae = mean_absolute_error(self.val_labels, predictions)
            r2 = r2_score(self.val_labels, predictions)

            self.epochs.append(epoch + 1)
            self.mse.append(mse)
            self.mae.append(mae)
            self.r2.append(r2)
            print(f'Epoch {epoch + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}')


metrics_callback = MetricsCallback(val_images, val_labels, epochs_to_evaluate=[10, 20, 30, 40, 50])

# Train the model
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs,
                    batch_size=batch_size, callbacks=[metrics_callback])

# Evaluate the model
val_loss = model.evaluate(val_images, val_labels)
print(f"Validation Loss: {val_loss}")

# Save the model
model.save("gaze_tracking_grayMixed_cnn50_02_.h5")

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('model_loss.png')
plt.show()

# Plot MSE, MAE, R-squared values
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(metrics_callback.epochs, metrics_callback.mse, marker='o', color='blue')
plt.title('Mean Squared Error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(metrics_callback.epochs, metrics_callback.mae, marker='o', color='green')
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(metrics_callback.epochs, metrics_callback.r2, marker='o', color='red')
plt.title('R-squared')
plt.xlabel('Epoch')
plt.ylabel('R-squared')
plt.grid(True)

plt.tight_layout()
plt.savefig('metrics.png')
plt.show()
