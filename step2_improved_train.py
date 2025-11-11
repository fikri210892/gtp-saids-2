#!/usr/bin/env python3
"""
step2_improved_train.py
Improved CNN training with better regularization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE

print("="*70)
print("  IMPROVED CNN TRAINING - Anti-Overfitting")
print("="*70)

# Load dataset
print("\n[*] Loading dataset...")
df = pd.read_csv('training_dataset.csv')

print(f"[+] Original dataset: {len(df)} samples")
print(f"    Normal (0): {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%)")
print(f"    Attack (1): {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)")

# Check class distribution
class_counts = df['label'].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\n[*] Class imbalance ratio: {imbalance_ratio:.2f}")

# Separate features and labels
feature_columns = [col for col in df.columns if col != 'label']
X = df[feature_columns].values
y = df['label'].values

# Clean data
print("[*] Cleaning data (NaN/Inf)...")
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Feature analysis
print("\n[*] Feature analysis:")
for i, col in enumerate(feature_columns):
    print(f"  {col:20s} - Range: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")

# Apply SMOTE for better balance (optional)
if imbalance_ratio > 1.5:
    print(f"\n[*] Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X, y = smote.fit_resample(X, y)
    print(f"[+] After SMOTE: {len(X)} samples")
    print(f"    Normal: {np.sum(y==0)}, Attack: {np.sum(y==1)}")

# Normalize
print("\n[*] Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split with stratification
print("[*] Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[+] Train: {len(X_train)} ({np.sum(y_train==0)} normal, {np.sum(y_train==1)} attack)")
print(f"[+] Test:  {len(X_test)} ({np.sum(y_test==0)} normal, {np.sum(y_test==1)} attack)")

# Build IMPROVED CNN Model
def build_improved_cnn(input_shape):
    """
    Improved CNN with stronger regularization
    - Simpler architecture (less layers)
    - Higher dropout rates
    - L2 regularization
    - Batch normalization
    """
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Conv Block 1 - Reduced filters
        layers.Conv1D(32, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),  # Increased from 0.3
        
        # Conv Block 2
        layers.Conv1D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.6),  # Increased from 0.3
        
        # Removed Conv Block 3 (simpler model)
        
        # Dense layers - Smaller units
        layers.Flatten(),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.6),  # Increased from 0.5
        
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.6),
        
        # Output
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Lower learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

print("\n[*] Building improved CNN model...")
model = build_improved_cnn((X_train.shape[1], 1))

print("\n" + "="*70)
print("IMPROVED MODEL ARCHITECTURE")
print("="*70)
model.summary()

# Callbacks with more aggressive early stopping
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'improved_cnn_model.h5',
        save_best_only=True,
        monitor='val_auc',  # Use AUC instead of accuracy
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
]

# Train with validation split
print("\n[*] Training improved CNN model...")
print("="*70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # More epochs, but early stopping will prevent overfitting
    batch_size=32,  # Smaller batch size
    callbacks=callbacks,
    class_weight={0: 1.0, 1: 1.0},  # Equal weights
    verbose=1
)

# Evaluate
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

results = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Results:")
print(f"  Loss:      {results[0]:.4f}")
print(f"  Accuracy:  {results[1]:.4f}")
print(f"  Precision: {results[2]:.4f}")
print(f"  Recall:    {results[3]:.4f}")
print(f"  AUC:       {results[4]:.4f}")

# Predictions
print("\n[*] Analyzing predictions...")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Detailed metrics
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, 
                          target_names=['Normal', 'Attack'],
                          digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

# Calculate rates
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\nFalse Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
print(f"False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")

# Prediction distribution analysis
print("\n" + "="*70)
print("PREDICTION SCORE DISTRIBUTION")
print("="*70)

normal_scores = y_pred_prob[y_test==0]
attack_scores = y_pred_prob[y_test==1]

print(f"\nNormal Traffic Predictions:")
print(f"  Mean:   {normal_scores.mean():.4f}")
print(f"  Median: {np.median(normal_scores):.4f}")
print(f"  Std:    {normal_scores.std():.4f}")
print(f"  Min:    {normal_scores.min():.4f}")
print(f"  Max:    {normal_scores.max():.4f}")

print(f"\nAttack Traffic Predictions:")
print(f"  Mean:   {attack_scores.mean():.4f}")
print(f"  Median: {np.median(attack_scores):.4f}")
print(f"  Std:    {attack_scores.std():.4f}")
print(f"  Min:    {attack_scores.min():.4f}")
print(f"  Max:    {attack_scores.max():.4f}")

# Check for overfitting indicators
print("\n" + "="*70)
print("OVERFITTING CHECK")
print("="*70)

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

acc_gap = abs(train_acc - val_acc)
loss_gap = abs(train_loss - val_loss)

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Val Accuracy:   {val_acc:.4f}")
print(f"Gap:            {acc_gap:.4f}")

print(f"\nTrain Loss: {train_loss:.4f}")
print(f"Val Loss:   {val_loss:.4f}")
print(f"Gap:        {loss_gap:.4f}")

if acc_gap < 0.05 and loss_gap < 0.1:
    print("\n✅ Model appears well-generalized (low overfitting)")
elif acc_gap < 0.10 and loss_gap < 0.2:
    print("\n⚠️  Model shows slight overfitting (acceptable)")
else:
    print("\n❌ Model is overfitting (needs more regularization)")

# Save model
print("\n[*] Saving improved model...")
model.save('improved_cnn_model.h5')
with open('improved_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("[+] Model saved: improved_cnn_model.h5")
print("[+] Scaler saved: improved_scaler.pkl")

# Enhanced Visualization
print("\n[*] Generating enhanced plots...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss curves
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: AUC curves
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(history.history['auc'], label='Train AUC', linewidth=2)
ax3.plot(history.history['val_auc'], label='Val AUC', linewidth=2)
ax3.set_title('Model AUC', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('AUC')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix
ax4 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            cbar_kws={'label': 'Count'})
ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax4.set_ylabel('True Label')
ax4.set_xlabel('Predicted Label')

# Plot 5: Prediction Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', edgecolor='black')
ax5.hist(attack_scores, bins=50, alpha=0.6, label='Attack', color='red', edgecolor='black')
ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax5.set_title('Prediction Score Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('Prediction Score')
ax5.set_ylabel('Frequency')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Score comparison
ax6 = fig.add_subplot(gs[1, 2])
box_data = [normal_scores.flatten(), attack_scores.flatten()]
bp = ax6.boxplot(box_data, labels=['Normal', 'Attack'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax6.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax6.set_title('Score Distribution by Class', fontsize=12, fontweight='bold')
ax6.set_ylabel('Prediction Score')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Learning rate
ax7 = fig.add_subplot(gs[2, 0])
if 'lr' in history.history:
    ax7.plot(history.history['lr'], linewidth=2, color='purple')
    ax7.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Learning Rate')
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3)
else:
    ax7.text(0.5, 0.5, 'Learning rate not logged', 
             ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')

# Plot 8: Precision-Recall
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(history.history['precision'], label='Train Precision', linewidth=2)
ax8.plot(history.history['val_precision'], label='Val Precision', linewidth=2)
ax8.plot(history.history['recall'], label='Train Recall', linewidth=2, linestyle='--')
ax8.plot(history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
ax8.set_title('Precision & Recall', fontsize=12, fontweight='bold')
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Score')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# Plot 9: Summary metrics
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
summary_text = f"""
FINAL METRICS

Accuracy:  {results[1]:.4f}
Precision: {results[2]:.4f}
Recall:    {results[3]:.4f}
AUC:       {results[4]:.4f}

FPR: {fpr:.4f}
FNR: {fnr:.4f}

Train-Val Gap:
  Acc:  {acc_gap:.4f}
  Loss: {loss_gap:.4f}
"""
ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.3))

plt.suptitle('Improved CNN Model - Training Results', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('improved_training_results.png', dpi=300, bbox_inches='tight')
print("[+] Enhanced plot saved: improved_training_results.png")

print("\n" + "="*70)
print("[✓] IMPROVED CNN TRAINING COMPLETED!")
print("="*70)
print("\nGenerated files:")
print("  - improved_cnn_model.h5 (Improved model)")
print("  - improved_scaler.pkl (Feature scaler)")
print("  - improved_training_results.png (Visualization)")
print("\nNext step:")
print("  python3 step3_hybrid_detection.py gtp_attack.pcap")
print("  (Update script to use improved_cnn_model.h5)")
print("="*70)
