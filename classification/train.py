
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = "pose_dataset.csv"
MODEL_DIR = "models_c"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(CSV_PATH)

y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# -----------------------------
# Encode labels
# -----------------------------
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Data Augmentation (for few-shot)
# -----------------------------
def augment_pose(X, y):
    # Small Gaussian noise
    noise = np.random.normal(0, 0.01, X.shape)
    X_aug = X + noise

    # Horizontal flip (assuming keypoints are [x1,y1,x2,y2,...])
    X_flip = X.copy()
    X_flip[:, ::2] *= -1  # flip x coordinates

    # Combine original + augmented
    X_combined = np.vstack([X, X_aug, X_flip])
    y_combined = np.concatenate([y, y, y])

    return X_combined, y_combined

X, y = augment_pose(X, y)

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
num_classes = len(np.unique(y))

# -----------------------------
# Build smaller adaptive model
# -----------------------------
model = tf.keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.8),

    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.8),

    layers.Dense(num_classes, activation="softmax")
])

# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3,
        patience=7,
        min_lr=1e-5
    )
]

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=1500,
    batch_size=16,
    # callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", acc)

# -----------------------------
# Save model and encoders
# -----------------------------
model.save(os.path.join(MODEL_DIR, "tennis_pose_classifier.h5"))
pickle.dump(label_enc, open(os.path.join(MODEL_DIR, "labels.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))

print("Saved:")
print(" - tennis_pose_classifier.h5")
print(" - labels.pkl")
print(" - scaler.pkl")



