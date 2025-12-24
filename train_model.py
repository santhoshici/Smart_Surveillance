import numpy as np
import pandas as pd
import json
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# TRAINING SCRIPT FOR SUSPICIOUS ACTIVITY DETECTION MODEL
# ============================================================================
# This script trains a model using 34 features (17 MediaPipe pose landmarks × 2)
# ============================================================================


TRAIN_DATA_PATH = "activity_data.csv"
MODEL_PATH = "suspicious_activity_model.h5"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_SIZE = 34

print("=" * 80)
print("SUSPICIOUS ACTIVITY DETECTION - MODEL TRAINING")
print("=" * 80)

print("\n[1/6] Loading training data...")
try:
    df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"✓ Loaded {len(df)} samples")
except FileNotFoundError:
    print(f"✗ Error: {TRAIN_DATA_PATH} not found")
    print("Please ensure your training data CSV is in the current directory")
    exit(1)

print("\n[2/6] Extracting features...")
feature_columns = [col for col in df.columns if col.startswith("feature_")]
X = df[feature_columns].values[:, :FEATURE_SIZE]  # Use only first 34 features
y = df["label"].values

print(f"✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
print(f"✓ Classes: {np.unique(y)}")

print("\n[3/6] Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

label_map = {int(i): label for i, label in enumerate(label_encoder.classes_)}
print(f"✓ Label map: {label_map}")

print("\n[4/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training set shape: {X_train_scaled.shape}")
print(f"✓ Test set shape: {X_test_scaled.shape}")

print("\n[5/6] Building neural network model...")
model = keras.Sequential(
    [
        layers.Input(shape=(FEATURE_SIZE,)),
        layers.Dense(128, activation="relu", name="dense_1"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", name="dense_2"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", name="dense_3"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(len(label_encoder.classes_), activation="softmax", name="output"),
    ],
    name="SkeletonActivityClassifier",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(model.summary())

print("\n[6/6] Training model...")
print("-" * 80)
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
        ),
    ],
)

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"✓ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✓ Test Loss: {test_loss:.4f}")

print("\nPer-class evaluation:")
y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
for i, class_name in enumerate(label_encoder.classes_):
    mask = y_test == i
    if mask.sum() > 0:
        class_accuracy = (y_pred[mask] == i).mean()
        print(f"  {class_name}: {class_accuracy * 100:.2f}% ({mask.sum()} samples)")

print("\n" + "=" * 80)
print("SAVING ARTIFACTS")
print("=" * 80)

print(f"Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print(f"✓ Model saved")

print(f"Saving scaler to {SCALER_PATH}...")
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved")

print(f"Saving label encoder to {LABEL_ENCODER_PATH}...")
with open(LABEL_ENCODER_PATH, "w") as f:
    json.dump(label_map, f, indent=2)
print(f"✓ Label encoder saved")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("1. Update your backend.py with the correct file paths if needed")
print("2. Run: python backend_final.py")
print("3. Open Streamlit frontend: streamlit run frontend.py")
print("\nYour model is ready for inference!")
