import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc 
)
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV


print("Loading models and test data...")
X_test, y_test = joblib.load('training/test_data.pkl')
svm_model = joblib.load('training/svm_model.pkl')
rf_model = joblib.load('training/rf_model.pkl')


calibrated_svm = CalibratedClassifierCV(svm_model, cv='prefit')
calibrated_svm.fit(X_test, y_test)


class ManualEnsemble:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    def predict_proba(self, X):
        return (self.m1.predict_proba(X) + self.m2.predict_proba(X)) / 2

    def predict(self, X):
        import numpy as np
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

ensemble = ManualEnsemble(calibrated_svm, rf_model)

def evaluate_model(name, model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='weighted')
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(classification_report(y, preds, target_names=['FAKE', 'REAL']))
    return preds

svm_preds  = evaluate_model('SVM', calibrated_svm, X_test, y_test)
rf_preds   = evaluate_model('Random Forest', rf_model, X_test, y_test)
ens_preds  = evaluate_model('Ensemble', ensemble, X_test, y_test)

def save_confusion_matrix(name, model, X, y, filename):
    preds = model.predict(X)
    disp = ConfusionMatrixDisplay.from_predictions(
        y, preds,
        display_labels=['FAKE', 'REAL'],
        cmap='Blues'
    )
    disp.ax_.set_title(f'Confusion Matrix — {name}')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

save_confusion_matrix('SVM',      calibrated_svm, X_test, y_test, 'training/confusion_matrix_svm.png')
save_confusion_matrix('Random Forest', rf_model,  X_test, y_test, 'training/confusion_matrix_rf.png')
save_confusion_matrix('Ensemble', ensemble,        X_test, y_test, 'training/confusion_matrix_ensemble.png')


plt.figure(figsize=(8, 6))

for name, model in [('SVM', calibrated_svm), ('Random Forest', rf_model), ('Ensemble', ensemble)]:
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Model Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('training/roc_curve.png', bbox_inches='tight')
plt.close()
print("Saved: training/roc_curve.png")

print("\nEvaluation complete.")