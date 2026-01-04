import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# CONFIGURATION
# =========================
# MODEL_PATH = r"C:\Users\Talha\OneDrive\Desktop\ML_Project\models\eye_cnn.h5"
# TEST_DATA_DIR = r"C:\Users\Talha\OneDrive\Desktop\ML_Project\train"  # or separate test folder


basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(basedir)
MODEL_PATH = os.path.join(project_root, 'models/eye_cnn.h5')
TEST_DATA_DIR = os.path.join(project_root, 'train')



MODEL_NAME = "Eye Drowsiness Model"
TRAINING_IMAGE_SIZE = (96, 96)  # MUST match the training size
BATCH_SIZE = 32

# =========================
# OUTPUT DIRECTORY
# =========================
output_dir = r"C:\Users\Talha\OneDrive\Desktop\ML_Project\result"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print(f"\n{'='*60}")
print(f"🔍 EVALUATING:{MODEL_NAME}")
print(f"{'='*60}\n")

print(f"📦 Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# =========================
# PREPARE TEST DATA
# =========================
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=TRAINING_IMAGE_SIZE,   # Match training input
    color_mode='grayscale',
    class_mode='binary',
    subset='validation',                # Use validation split
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"\nTest samples: {test_generator.samples}")
print(f"Classes: {list(test_generator.class_indices.keys())}\n")

# =========================
# GENERATE PREDICTIONS
# =========================
print("🔮 Generating predictions...")
test_generator.reset()
y_true = test_generator.classes
y_pred_proba = model.predict(test_generator, verbose=1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# =========================
# CALCULATE METRICS
# =========================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_true, y_pred_proba)
except:
    auc = 0.0

# =========================
# DISPLAY RESULTS
# =========================
print(f"\n{'='*60}")
print(f"🎯 EVALUATION RESULTS")
print(f"{'='*60}\n")

print(f"📊 CORE METRICS:")
print(f"   ├─ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   ├─ Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   ├─ Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   ├─ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"   └─ AUC:       {auc:.4f}")

class_names = list(test_generator.class_indices.keys())
print(f"\n📋 CLASSIFICATION REPORT:\n{classification_report(y_true, y_pred, target_names=class_names, digits=4)}")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"🔢 CONFUSION MATRIX:")
print(f"              Predicted")
print(f"              {class_names[0]:<12} {class_names[1]:<12}")
print(f"Actual {class_names[0]:<7} {cm[0,0]:<12} {cm[0,1]:<12}")
print(f"       {class_names[1]:<7} {cm[1,0]:<12} {cm[1,1]:<12}")
print(f"\n🎲 ADDITIONAL METRICS:")
print(f"   ├─ True Positives:  {tp}")
print(f"   ├─ True Negatives:  {tn}")
print(f"   ├─ False Positives: {fp}")
print(f"   ├─ False Negatives: {fn}")
print(f"   └─ Specificity:     {specificity:.4f}")

# =========================
# VISUALIZATIONS
# =========================

# Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 16, 'weight': 'bold'})
plt.title(f'Confusion Matrix - {MODEL_NAME}', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{MODEL_NAME.replace(" ", "_")}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Confusion matrix saved!")

# Metrics Bar Chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
metrics_values = [accuracy, precision, recall, f1, auc]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D']

plt.figure(figsize=(12, 7))
bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
plt.ylim(0, 1.1)
plt.title(f'Performance Metrics - {MODEL_NAME}', fontsize=18, fontweight='bold')
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')

for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.4f}\n({value*100:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (80%)')
plt.axhline(y=0.9, color='darkgreen', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (90%)')
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{MODEL_NAME.replace(" ", "_")}_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Metrics chart saved!")

# ROC Curve
if auc > 0:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {MODEL_NAME}', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{MODEL_NAME.replace(" ", "_")}_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved!")

# Save results to text file
results_file = os.path.join(output_dir, f'{MODEL_NAME.replace(" ", "_")}_evaluation_results.txt')
with open(results_file, 'w') as f:
    f.write(f"{'='*60}\nEVALUATION RESULTS - {MODEL_NAME}\n{'='*60}\n\n")
    f.write(f"Model Path: {MODEL_PATH}\n")
    f.write(f"Test Data: {TEST_DATA_DIR}\n")
    f.write(f"Test Samples: {test_generator.samples}\n\n")
    f.write(f"CORE METRICS:\n")
    f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"  Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"  AUC:       {auc:.4f}\n\n")
    f.write(f"CONFUSION MATRIX:\n")
    f.write(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n\n")
    f.write(f"CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))

print(f"✅ Results saved to text file!\n")
print(f"{'='*60}")
print(f"✅ Evaluation complete! All results saved to: {output_dir}")
print(f"{'='*60}\n")
