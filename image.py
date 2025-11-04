# image_classification_tf.py
# ----------------------------
# Deep Learning Model for Image Classification
# using TensorFlow and Keras
# ----------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =============================
# 1️⃣ Load and Explore Dataset
# =============================
# We'll use CIFAR-10 dataset (10 categories of images)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values (0-255 → 0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Class names for visualization
class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# =============================
# 2️⃣ Visualize Sample Images
# =============================
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.suptitle("Sample CIFAR-10 Images")
plt.show()

# =============================
# 3️⃣ Build CNN Model
# =============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Model summary
model.summary()

# =============================
# 4️⃣ Compile the Model
# =============================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# =============================
# 5️⃣ Train the Model
# =============================
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test))

# =============================
# 6️⃣ Evaluate the Model
# =============================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc:.3f}")

# =============================
# 7️⃣ Visualize Training Results
# =============================
plt.figure(figsize=(10,4))

# Accuracy graph
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss graph
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# =============================
# 8️⃣ Predict and Show Results
# =============================
import numpy as np

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    img = X_test[i]
    plt.imshow(img)
    pred = model.predict(img.reshape(1,32,32,3))
    predicted_label = class_names[np.argmax(pred)]
    true_label = class_names[y_test[i][0]]
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.axis('off')

plt.suptitle("Model Predictions vs Actual Labels")
plt.show()
