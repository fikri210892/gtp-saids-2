#!/usr/bin/env python3
"""
step2_train_cnn.py
Train CNN model untuk GTP anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("[!] TensorFlow not installed!")
    print("[*] Installing: pip install tensorflow")
    import sys
    sys.exit(1)

print("="*60)
print("  CNN Model Training for GTP Attack Detection")
print("="*60)

# Load dataset
print("\n[*] Loading dataset...")
df = pd.read_csv('training_dataset.csv')

print(f"[+] Dataset loaded: {len(df)} samples")
print(f"    Normal (0): {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%)")
print(f"    Attack (1): {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)")

# Separate features and labels
feature_columns = [col for col in df.columns if col != 'label']
X = df[feature_columns].values
y = df['label'].values

print(f"[+] Features: {X.shape[1]}")

# Check for NaN/Inf
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    print("[*] Cleaning NaN/Inf values...")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize features
print("[*] Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN (samples, features, channels)
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split dataset
print("[*] Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[+] Train samples: {len(X_train)}")
print(f"[+] Test samples:  {len(X_test)}")

# Build CNN Model
def build_cnn_model(input_shape):
    """Build 1D CNN model"""
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Conv Block 1
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Conv Block 2
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Conv Block 3
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

print("\n[*] Building CNN model...")
model = build_cnn_model((X_train.shape[1], 1))

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
model.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_cnn_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

# Train
print("\n[*] Training CNN model...")
print("="*60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

results = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Results:")
print(f"  Loss:      {results[0]:.4f}")
print(f"  Accuracy:  {results[1]:.4f}")
print(f"  Precision: {results[2]:.4f}")
print(f"  Recall:    {results[3]:.4f}")
print(f"  AUC:       {results[4]:.4f}")

# Predictions
print("\n[*] Generating predictions...")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save model and scaler
print("\n[*] Saving model and scaler...")
model.save('gtp_cnn_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("[+] Model saved: gtp_cnn_model.h5")
print("[+] Scaler saved: scaler.pkl")

# Plot training history
print("\n[*] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Loss
axes[0, 1].plot(history.history['loss'], label='Train Loss')
axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# Plot 4: ROC-like (prediction distribution)
axes[1, 1].hist(y_pred_prob[y_test==0], bins=50, alpha=0.5, label='Normal', color='blue')
axes[1, 1].hist(y_pred_prob[y_test==1], bins=50, alpha=0.5, label='Attack', color='red')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
axes[1, 1].set_title('Prediction Score Distribution')
axes[1, 1].set_xlabel('Prediction Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300)
print("[+] Training plot saved: training_results.png")

print("\n" + "="*60)
print("[✓] CNN Training Completed!")
print("="*60)
print("\nGenerated files:")
print("  - gtp_cnn_model.h5 (CNN model)")
print("  - scaler.pkl (Feature scaler)")
print("  - training_results.png (Plots)")
print("\nNext step: python3 step3_hybrid_detection.py <pcap_file>")
print("="*60)
