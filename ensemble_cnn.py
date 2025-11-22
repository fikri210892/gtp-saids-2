#!/usr/bin/env python3
"""
step2_ensemble_cnn_train.py
Ensemble CNN Training for GTP IDS - NOVELTY APPROACH
Multiple diverse CNN architectures combined via voting for improved detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  ENSEMBLE CNN TRAINING - NOVELTY APPROACH")
print("  Multiple CNN Architectures with Voting Mechanism")
print("="*70)

# Load dataset
print("\n[*] Loading dataset...")
df = pd.read_csv('training_dataset_fixed.csv')

print(f"[+] Dataset: {len(df)} samples")
print(f"    Normal (0): {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%)")
print(f"    Attack (1): {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)")

# Check imbalance
imbalance_ratio = len(df[df['label']==0]) / len(df[df['label']==1])
print(f"\n[*] Class imbalance ratio: {imbalance_ratio:.2f}:1")

# Separate features and labels
feature_columns = [col for col in df.columns if col != 'label']
X = df[feature_columns].values
y = df['label'].values

# Clean data
print("\n[*] Cleaning data...")
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Use RobustScaler
print("\n[*] Scaling features with RobustScaler...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Proper train/val/test split
print("\n[*] Creating train/val/test split (70/15/15)...")

# First split: 70% train, 30% temp
X_train_full, X_temp, y_train_full, y_temp = train_test_split(
    X_cnn, y, test_size=0.30, random_state=42, stratify=y
)

# Second split: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"[+] Train: {len(X_train_full)} samples")
print(f"[+] Val:   {len(X_val)} samples")
print(f"[+] Test:  {len(X_test)} samples")

# Compute class weights
print("\n[*] Computing class weights...")
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train_full),
                                     y=y_train_full)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"    Class 0 (Normal): {class_weight_dict[0]:.3f}")
print(f"    Class 1 (Attack): {class_weight_dict[1]:.3f}")

# Apply light label smoothing
print("\n[*] Applying light label smoothing (0.02)...")
y_train_smoothed = np.where(y_train_full == 1, 0.98, 0.02)
y_val_smoothed = np.where(y_val == 1, 0.98, 0.02)

# ========================================================================
# ENSEMBLE CNN MODELS - 5 DIVERSE ARCHITECTURES
# ========================================================================

print("\n" + "="*70)
print("BUILDING ENSEMBLE: 5 DIVERSE CNN ARCHITECTURES")
print("="*70)

def build_cnn_shallow_wide(input_shape):
    """
    Model 1: Shallow & Wide
    - Few layers, many filters
    - Fast learning, broad feature detection
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GaussianNoise(0.02),
        
        # Single wide conv block
        layers.Conv1D(128, 5, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.MaxPooling1D(2),
        
        # Dense
        layers.Flatten(),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='Shallow_Wide')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )
    return model

def build_cnn_deep_narrow(input_shape):
    """
    Model 2: Deep & Narrow
    - Many layers, fewer filters
    - Deep feature extraction
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GaussianNoise(0.02),
        
        # Conv Block 1
        layers.Conv1D(16, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.MaxPooling1D(2),
        
        # Conv Block 2
        layers.Conv1D(32, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.MaxPooling1D(2),
        
        # Conv Block 3
        layers.Conv1D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Dense
        layers.Flatten(),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='Deep_Narrow')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )
    return model

def build_cnn_balanced(input_shape):
    """
    Model 3: Balanced (Baseline)
    - Medium depth and width
    - General purpose
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GaussianNoise(0.02),
        
        # Conv Block 1
        layers.Conv1D(32, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.MaxPooling1D(2),
        
        # Conv Block 2
        layers.Conv1D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.MaxPooling1D(2),
        
        # Dense
        layers.Flatten(),
        layers.Dense(48, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(24, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name='Balanced')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )
    return model

def build_cnn_large_kernel(input_shape):
    """
    Model 4: Large Kernel Focus
    - Larger receptive field
    - Capture long-range patterns
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GaussianNoise(0.02),
        
        # Conv Block 1 - Large kernel
        layers.Conv1D(32, 7, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.MaxPooling1D(2),
        
        # Conv Block 2
        layers.Conv1D(64, 5, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.MaxPooling1D(2),
        
        # Dense
        layers.Flatten(),
        layers.Dense(48, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='Large_Kernel')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )
    return model

def build_cnn_small_kernel(input_shape):
    """
    Model 5: Small Kernel Focus
    - Fine-grained feature detection
    - Local pattern emphasis
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GaussianNoise(0.02),
        
        # Conv Block 1 - Small kernel
        layers.Conv1D(48, 2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.MaxPooling1D(2),
        
        # Conv Block 2
        layers.Conv1D(96, 2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.MaxPooling1D(2),
        
        # Dense
        layers.Flatten(),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='Small_Kernel')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )
    return model

# Build all models
model_builders = [
    ('Shallow_Wide', build_cnn_shallow_wide),
    ('Deep_Narrow', build_cnn_deep_narrow),
    ('Balanced', build_cnn_balanced),
    ('Large_Kernel', build_cnn_large_kernel),
    ('Small_Kernel', build_cnn_small_kernel)
]

print("\nEnsemble Models:")
for i, (name, _) in enumerate(model_builders, 1):
    print(f"  {i}. {name}")

# ========================================================================
# TRAIN EACH MODEL IN ENSEMBLE
# ========================================================================

print("\n" + "="*70)
print("TRAINING ENSEMBLE MODELS")
print("="*70)

trained_models = []
training_histories = []
model_results = []

for i, (model_name, build_func) in enumerate(model_builders, 1):
    print(f"\n{'='*70}")
    print(f"Training Model {i}/5: {model_name}")
    print(f"{'='*70}")
    
    # Build model
    model = build_func((X_train_full.shape[1], 1))
    
    # Show architecture
    print(f"\n{model_name} Architecture:")
    model.summary()
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    print(f"Total trainable parameters: {trainable_params:,}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=8,
            restore_best_weights=True,
            mode='max',
            verbose=0
        ),
        ModelCheckpoint(
            f'ensemble_model_{i}_{model_name}.h5',
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=0.00001,
            verbose=0
        )
    ]
    
    # Train
    print(f"\nTraining {model_name}...")
    history = model.fit(
        X_train_full, y_train_smoothed,
        validation_data=(X_val, y_val_smoothed),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc, test_precision, test_recall, test_auc = test_results
    
    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Store results
    trained_models.append(model)
    training_histories.append(history)
    model_results.append({
        'name': model_name,
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'auc': test_auc,
        'predictions': y_pred_prob,
        'params': trainable_params
    })
    
    print(f"\n{model_name} Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  AUC:       {test_auc:.4f}")

# ========================================================================
# ENSEMBLE VOTING
# ========================================================================

print("\n" + "="*70)
print("ENSEMBLE VOTING")
print("="*70)

# Collect all predictions
all_predictions = np.array([result['predictions'] for result in model_results])

# Soft Voting (Average probabilities)
print("\n[*] Computing Soft Voting (Average)...")
ensemble_pred_prob_soft = np.mean(all_predictions, axis=0)
ensemble_pred_soft = (ensemble_pred_prob_soft > 0.5).astype(int)

# Hard Voting (Majority vote on binary predictions)
print("[*] Computing Hard Voting (Majority)...")
all_binary_predictions = (all_predictions > 0.5).astype(int)
ensemble_pred_hard = np.round(np.mean(all_binary_predictions, axis=0)).astype(int)

# Weighted Voting (Weight by AUC)
print("[*] Computing Weighted Voting (by AUC)...")
weights = np.array([result['auc'] for result in model_results])
weights = weights / np.sum(weights)  # Normalize
ensemble_pred_prob_weighted = np.average(all_predictions, axis=0, weights=weights)
ensemble_pred_weighted = (ensemble_pred_prob_weighted > 0.5).astype(int)

print(f"\nWeights (by AUC):")
for name, weight in zip([r['name'] for r in model_results], weights):
    print(f"  {name:15s}: {weight:.4f}")

# ========================================================================
# EVALUATION
# ========================================================================

print("\n" + "="*70)
print("ENSEMBLE EVALUATION")
print("="*70)

def evaluate_predictions(y_true, y_pred, y_pred_prob, method_name):
    """Evaluate and print metrics"""
    from sklearn.metrics import f1_score
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n{method_name}:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC:       {auc:.4f}")
    print(f"  FPR:       {fpr_rate:.4f} ({fpr_rate*100:.2f}%)")
    print(f"  FNR:       {fnr_rate:.4f} ({fnr_rate*100:.2f}%)")
    print(f"  Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'fpr': fpr_rate,
        'fnr': fnr_rate,
        'cm': cm
    }

# Evaluate individual models
print("\n" + "="*70)
print("INDIVIDUAL MODEL PERFORMANCE")
print("="*70)

individual_metrics = []
for result in model_results:
    y_pred = (result['predictions'] > 0.5).astype(int)
    metrics = evaluate_predictions(y_test, y_pred, result['predictions'], result['name'])
    individual_metrics.append(metrics)

# Evaluate ensemble methods
print("\n" + "="*70)
print("ENSEMBLE METHODS COMPARISON")
print("="*70)

soft_metrics = evaluate_predictions(y_test, ensemble_pred_soft, ensemble_pred_prob_soft, "Soft Voting (Average)")
hard_metrics = evaluate_predictions(y_test, ensemble_pred_hard, ensemble_pred_prob_soft, "Hard Voting (Majority)")
weighted_metrics = evaluate_predictions(y_test, ensemble_pred_weighted, ensemble_pred_prob_weighted, "Weighted Voting (AUC)")

# ========================================================================
# VISUALIZATION
# ========================================================================

print("\n[*] Generating comprehensive visualization...")

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

# Plot 1: Individual Model Performance Comparison
ax1 = fig.add_subplot(gs[0, :2])
model_names = [r['name'] for r in model_results]
accuracies = [r['accuracy'] for r in model_results]
precisions = [r['precision'] for r in model_results]
recalls = [r['recall'] for r in model_results]
aucs = [r['auc'] for r in model_results]

x = np.arange(len(model_names))
width = 0.2

bars1 = ax1.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='steelblue', edgecolor='black')
bars2 = ax1.bar(x - 0.5*width, precisions, width, label='Precision', color='coral', edgecolor='black')
bars3 = ax1.bar(x + 0.5*width, recalls, width, label='Recall', color='lightgreen', edgecolor='black')
bars4 = ax1.bar(x + 1.5*width, aucs, width, label='AUC', color='gold', edgecolor='black')

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Individual Model Performance Comparison', fontsize=14, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.set_ylim([0, 1.05])
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 2: Ensemble vs Best Individual
ax2 = fig.add_subplot(gs[0, 2:])
methods = ['Best\nIndividual', 'Soft\nVoting', 'Hard\nVoting', 'Weighted\nVoting']
best_individual_acc = max(accuracies)
ensemble_accs = [best_individual_acc, soft_metrics['accuracy'], 
                 hard_metrics['accuracy'], weighted_metrics['accuracy']]

bars = ax2.bar(methods, ensemble_accs, color=['steelblue', 'coral', 'lightgreen', 'gold'], 
               edgecolor='black', linewidth=1.5, width=0.6)
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Ensemble vs Individual Performance', fontsize=14, fontweight='bold', pad=10)
ax2.set_ylim([0.85, 1.0])
ax2.set_yticks(np.arange(0.85, 1.05, 0.02))
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

for bar, acc in zip(bars, ensemble_accs):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{acc:.4f}\n({acc*100:.2f}%)', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

# Plot 3-7: Individual Model Confusion Matrices
for idx, (result, metrics) in enumerate(zip(model_results, individual_metrics)):
    row = 1 + idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_title(f'{result["name"]}\nAcc: {result["accuracy"]:.3f}', 
                fontsize=11, fontweight='bold', pad=5)
    ax.set_ylabel('True', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=10)

# Plot 8: Ensemble Soft Voting Confusion Matrix
ax8 = fig.add_subplot(gs[2, 3])
sns.heatmap(soft_metrics['cm'], annot=True, fmt='d', cmap='Greens', ax=ax8,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 12, 'weight': 'bold'})
ax8.set_title(f'Ensemble (Soft Voting)\nAcc: {soft_metrics["accuracy"]:.4f}', 
            fontsize=12, fontweight='bold', pad=5)
ax8.set_ylabel('True Label', fontsize=10)
ax8.set_xlabel('Predicted Label', fontsize=10)

# Plot 9: ROC Curves Comparison
ax9 = fig.add_subplot(gs[3, :2])
for result in model_results:
    fpr, tpr, _ = roc_curve(y_test, result['predictions'])
    ax9.plot(fpr, tpr, linewidth=2, label=f'{result["name"]} (AUC={result["auc"]:.4f})', alpha=0.7)

# Ensemble ROC
fpr_soft, tpr_soft, _ = roc_curve(y_test, ensemble_pred_prob_soft)
ax9.plot(fpr_soft, tpr_soft, linewidth=3, label=f'Ensemble Soft (AUC={soft_metrics["auc"]:.4f})', 
         color='red', linestyle='--')

ax9.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
ax9.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax9.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax9.set_title('ROC Curves: Individual Models vs Ensemble', fontsize=14, fontweight='bold', pad=10)
ax9.set_xlim([-0.02, 1.02])
ax9.set_ylim([-0.02, 1.02])
ax9.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax9.grid(True, alpha=0.3, linestyle='--')

# Plot 10: Model Parameters vs Performance
ax10 = fig.add_subplot(gs[3, 2])
params = [r['params'] for r in model_results]
scatter = ax10.scatter(params, accuracies, s=200, c=aucs, cmap='viridis', 
                      edgecolor='black', linewidth=2, alpha=0.8)
for i, name in enumerate(model_names):
    ax10.annotate(name, (params[i], accuracies[i]), 
                 textcoords="offset points", xytext=(0,10), 
                 ha='center', fontsize=8)
ax10.set_xlabel('Parameters Count', fontsize=11, fontweight='bold')
ax10.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax10.set_title('Model Complexity vs Performance', fontsize=12, fontweight='bold', pad=10)
ax10.grid(True, alpha=0.3, linestyle='--')
plt.colorbar(scatter, ax=ax10, label='AUC')

# Plot 11: Summary
ax11 = fig.add_subplot(gs[3, 3])
ax11.axis('off')
summary_text = f"""ENSEMBLE SUMMARY

Best Individual Model:
  {model_results[np.argmax(accuracies)]['name']}
  Accuracy: {max(accuracies):.4f}
  AUC:      {max(aucs):.4f}

Ensemble (Soft Voting):
  Accuracy:  {soft_metrics['accuracy']:.4f}
  Precision: {soft_metrics['precision']:.4f}
  Recall:    {soft_metrics['recall']:.4f}
  F1-Score:  {soft_metrics['f1']:.4f}
  AUC:       {soft_metrics['auc']:.4f}

Improvement:
  Acc Gain: {(soft_metrics['accuracy']-max(accuracies))*100:+.2f}%
  
FPR: {soft_metrics['fpr']*100:.2f}%
FNR: {soft_metrics['fnr']*100:.2f}%

Total Models: 5
Voting: Soft (Average)
"""
ax11.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=1.5))

plt.suptitle('Ensemble CNN Training Results - 5 Diverse Architectures', 
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('ensemble_training_results.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[+] Visualization saved: ensemble_training_results.png")

# Save models and scaler
print("\n[*] Saving models and scaler...")
for i, (model, (model_name, _)) in enumerate(zip(trained_models, model_builders), 1):
    model.save(f'ensemble_model_{i}_{model_name}.h5')
    print(f"[+] Saved: ensemble_model_{i}_{model_name}.h5")

with open('ensemble_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("[+] Scaler saved: ensemble_scaler.pkl")

# Save ensemble predictions for later use
np.save('ensemble_predictions.npy', all_predictions)
np.save('ensemble_weights.npy', weights)
print("[+] Ensemble predictions and weights saved")

# ========================================================================
# FINAL SUMMARY
# ========================================================================

print("\n" + "="*70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)

print("\nGenerated Files:")
print("  - ensemble_model_1_Shallow_Wide.h5")
print("  - ensemble_model_2_Deep_Narrow.h5")
print("  - ensemble_model_3_Balanced.h5")
print("  - ensemble_model_4_Large_Kernel.h5")
print("  - ensemble_model_5_Small_Kernel.h5")
print("  - ensemble_scaler.pkl")
print("  - ensemble_predictions.npy")
print("  - ensemble_weights.npy")
print("  - ensemble_training_results.png")

print("\nIndividual Model Performance:")
for result in model_results:
    print(f"  {result['name']:15s}: Acc={result['accuracy']:.4f}, AUC={result['auc']:.4f}")

print("\nEnsemble Performance:")
print(f"  Soft Voting:     Acc={soft_metrics['accuracy']:.4f}, AUC={soft_metrics['auc']:.4f}")
print(f"  Hard Voting:     Acc={hard_metrics['accuracy']:.4f}, AUC={hard_metrics['auc']:.4f}")
print(f"  Weighted Voting: Acc={weighted_metrics['accuracy']:.4f}, AUC={weighted_metrics['auc']:.4f}")

improvement = (soft_metrics['accuracy'] - max(accuracies)) * 100
print(f"\nEnsemble Improvement: {improvement:+.2f}%")

if soft_metrics['accuracy'] > max(accuracies):
    print("\n[✓✓✓] ENSEMBLE OUTPERFORMS INDIVIDUAL MODELS!")
elif soft_metrics['accuracy'] >= max(accuracies) - 0.005:
    print("\n[✓✓] ENSEMBLE COMPARABLE TO BEST INDIVIDUAL MODEL")
else:
    print("\n[✓] INDIVIDUAL MODEL PERFORMS BETTER")

print("\nNext Steps:")
print("  1. Review ensemble_training_results.png")
print("  2. Compare with baseline single CNN model")
print("  3. Integrate into step3_hybrid_detection.py")
print("  4. Test ensemble on real GTP traffic")
print("="*70)
