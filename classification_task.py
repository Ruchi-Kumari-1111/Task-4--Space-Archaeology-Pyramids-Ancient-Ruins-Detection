import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, 
    f1_score, roc_curve, auc, precision_recall_curve
)

# Define your exact target directory
TARGET_DIR = r"D:\VS codes\Internship(7thSem)\Task-4"

# Create the directory if it doesn't exist yet
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print(f"--> Created missing directory: {TARGET_DIR}")

print("="*70)
print("   STARTING SPACE ARCHAEOLOGY: SATELLITE ANOMALY CLASSIFIER")
print("="*70)

# ==========================================
# STEP 1: CHOOSE / GENERATE THE BINARY DATASET
# ==========================================
print("\n[STEP 1] Initializing Space Archaeology Dataset...")
np.random.seed(42)
n_samples = 5000 

y = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

ndvi = np.where(y == 1, np.random.normal(0.35, 0.1, n_samples), np.random.normal(0.70, 0.08, n_samples))
elevation = np.where(y == 1, np.random.normal(2.8, 0.6, n_samples), np.random.normal(0.6, 0.15, n_samples))
radar = np.where(y == 1, np.random.normal(-10.0, 1.8, n_samples), np.random.normal(-18.0, 1.2, n_samples))

df = pd.DataFrame({
    'NDVI_Index': ndvi,
    'Elevation_Variance': elevation,
    'Radar_Backscatter': radar,
    'Contains_Ruin': y
})

# SAVE THE CSV FILE TO TARGET DIRECTORY
csv_path = os.path.join(TARGET_DIR, "space_archaeology_dataset.csv")
df.to_csv(csv_path, index=False)
print(f"--> Dataset Saved Successfully to: {csv_path}")
print(f"--> Class Distribution:\n{df['Contains_Ruin'].value_counts()}")

# ==========================================
# STEP 2: TRAIN/TEST SPLIT AND STANDARDIZE
# ==========================================
print("\n[STEP 2] Splitting and Standardizing Features...")
X = df.drop('Contains_Ruin', axis=1)
y = df['Contains_Ruin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# STEP 3: FIT A LOGISTIC REGRESSION MODEL
# ==========================================
print("\n[STEP 3] Fitting Logistic Regression Model...")
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_probs = model.predict_proba(X_test_scaled)[:, 1]
y_pred_default = (y_probs >= 0.5).astype(int)

# ==========================================
# STEP 4 & 5: PLOTTING, SAVING, AND EXPLAINING GRAPHS
# ==========================================
print("\n" + "="*50)
print("             GENERATING & SAVING GRAPHICAL ANALYSIS")
print("="*50)

# --- GRAPH 1: THE LOGISTIC REGRESSION MODEL (SIGMOID CURVE) ---
plt.figure(figsize=(8, 5))
log_odds = model.decision_function(X_test_scaled)
sorted_indices = np.argsort(log_odds)

plt.scatter(log_odds[sorted_indices], y_test.iloc[sorted_indices], color='darkorange', alpha=0.4, label='Actual Ground Truth')
plt.plot(log_odds[sorted_indices], y_probs[sorted_indices], color='navy', linewidth=3, label='Sigmoid Curve')
plt.axhline(0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
plt.title('Logistic Regression Sigmoid Curve (Space Archaeology Context)')
plt.xlabel('Log-Odds Linearly Combined (β₀ + β₁X₁ + ...)')
plt.ylabel('Predicted Probability P(Ruin = 1)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save Graph 1
g1_path = os.path.join(TARGET_DIR, "1_sigmoid_curve.png")
plt.savefig(g1_path, dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 GRAPH 1 SAVED TO:", g1_path)
print("--- GRAPH 1 EXPLANATION: THE LOGISTIC REGRESSION (SIGMOID CURVE) GRAPH ---")
print("• What is happening:")
print("  This graph maps out the raw geometric distances (log-odds) into bounded probabilities.")
print("  The orange dots at y=0 represent coordinates with nothing there, and y=1 are coordinates with pyramids.")
print("  The dark blue line represents the Sigmoid Function: P(Y=1|X) = 1 / (1 + e^-z).")
print("• Theoretical Insight:")
print("  Linear regression allows unbounded outputs (-inf to +inf). The Sigmoid function forcefully squeezes")
print("  any linear regression line output into an elegant, smooth 'S' curve between exactly 0 and 1.")
print("  Any point where the curve crosses above the red dashed line (0.5) is default-classified as a Ruin.")
print("-"*85)


# --- GRAPH 2: CONFUSION MATRIX ---
cm = confusion_matrix(y_test, y_pred_default)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Terrain (0)', 'Ruin (1)'], yticklabels=['Terrain (0)', 'Ruin (1)'])
plt.title('Confusion Matrix (Default 0.5 Threshold)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Save Graph 2
g2_path = os.path.join(TARGET_DIR, "2_confusion_matrix.png")
plt.savefig(g2_path, dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 GRAPH 2 SAVED TO:", g2_path)
print("--- GRAPH 2 EXPLANATION: THE CONFUSION MATRIX ---")
print("• What is happening (At default 0.5 Threshold):")
print(f"  - True Negatives (TN): {cm[0,0]} -> Correctly verified empty natural terrain.")
print(f"  - False Positives (FP): {cm[0,1]} -> False alarms! Empty terrain flagged as ruins.")
print(f"  - False Negatives (FN): {cm[1,0]} -> Disaster! Hidden ruins missed by the model.")
print(f"  - True Positives (TP): {cm[1,1]} -> Successfully found an ancient archaeological site!")
print("• Theoretical Insight:")
print("  Because space archaeology is an extremely imbalanced 'needle-in-a-haystack' task, looking at")
print("  raw accuracy is deceptive. This matrix reveals exactly where the model is stumbling.")
print("-"*85)


# --- GRAPH 3: ROC-AUC CURVE ---
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Save Graph 3
g3_path = os.path.join(TARGET_DIR, "3_roc_auc_curve.png")
plt.savefig(g3_path, dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 GRAPH 3 SAVED TO:", g3_path)
print("--- GRAPH 3 EXPLANATION: THE ROC-AUC CURVE ---")
print("• What is happening:")
print(f"  The green line tracks how the True Positive Rate behaves against the False Positive Rate as we alter")
print(f"  the prediction boundary. Your current Area Under the Curve (AUC) is: {roc_auc:.2f}.")
print("• Theoretical Insight:")
print("  An AUC of 0.5 (the grey diagonal line) is completely useless random guessing. An AUC of 1.0 is perfect.")
print("  Our exceptionally high score proves the remote-sensing physics features (LiDAR, Radar, NDVI)")
print("  provide an extremely strong mathematical signal for isolating ruins, despite severe class imbalance.")
print("-"*85)


# --- GRAPH 4: THRESHOLD TUNING VS PRECISION, RECALL & F1 ---
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], 'b--', label='Precision (Accuracy of alarms)', lw=2)
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall (Sensitivity/Catch rate)', lw=2)
plt.plot(thresholds, f1_scores, 'r:', label='F1-Score (Harmonic Balance)', lw=2)
plt.title('Threshold Tuning: Precision, Recall, and F1-Score vs Decision Threshold')
plt.xlabel('Probability Threshold Selection')
plt.ylabel('Metric Score Metric')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)

# Save Graph 4
g4_path = os.path.join(TARGET_DIR, "4_threshold_tuning.png")
plt.savefig(g4_path, dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 GRAPH 4 SAVED TO:", g4_path)
print("--- GRAPH 4 EXPLANATION: THRESHOLD VALUE & METRIC TUNING GRAPH ---")
print("• What is happening:")
print("  This graph tracks Precision (blue line) and Recall (green line) across every single choice")
print("  of custom probability thresholds from 0.0 to 1.0.")
print("• Theoretical Insight & The Operational Trade-Off:")
print("  - If you set a LOW threshold (e.g., 0.2): Recall skyrockets to near 100%. You will catch EVERY single")
print("    ruin on Earth, but your precision plummets because you'll launch expensive field operations out to")
print("    barren hills that are just natural rock anomalies (High False Positives).")
print("  - If you set a HIGH threshold (e.g., 0.8): Precision climbs to near 100%. When the model sounds an alarm,")
print("    you are completely guaranteed a ruin is there. But your Recall crashes—leaving dozens of hidden")
print("    pyramids completely undetected because the model wasn't 100% confident.")
print("  - The intersection or peak of the red dotted line (F1-Score) represents the optimal balance.")
print("-"*85)

# Final summary printout
print("\n[SUMMARY] Standard Technical Deliverables Evaluation Metrics (Threshold = 0.5):")
print(f"  - Baseline Precision: {precision_score(y_test, y_pred_default):.2f}")
print(f"  - Baseline Recall:    {recall_score(y_test, y_pred_default):.2f}")
print(f"  - Baseline F1-Score:  {f1_score(y_test, y_pred_default):.2f}")
print("="*70)
print(f"SUCCESS: All files successfully written to {TARGET_DIR}")
print("="*70)