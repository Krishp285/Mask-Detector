# Face Mask Detector by [Your Name]
import tensorflow as tf
import os

# Paths
train_dir = "C:/mask_detector_project/Face Mask Dataset/Train"
test_dir = "C:/mask_detector_project/Face Mask Dataset/Test"

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
).map(lambda x, y: (x / 255.0, y))

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
).map(lambda x, y: (x / 255.0, y))

# Build CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Save
model.save("mask_detector_model.h5")