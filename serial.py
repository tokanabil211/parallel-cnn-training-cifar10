import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import numpy as np
import time

# Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train and time
start_time = time.time()
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=2)
end_time = time.time()

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(x_test), axis=1)

print(f"\n⏱️ Serial Training Time: {end_time - start_time:.2f} seconds")
print(f"✅ Final Test Accuracy: {test_acc:.4f}")
print(f"✅ Final Test Loss: {test_loss:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
