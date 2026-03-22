import joblib
import numpy as np
import os
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'training')

tfidf     = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
svm_model = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
rf_model  = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))



calibrated_svm = joblib.load(os.path.join(MODEL_DIR, 'calibrated_svm.pkl'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

def predict(text):
    cleaned  = clean_text(text)
    vector   = tfidf.transform([cleaned])

    svm_pred  = calibrated_svm.predict(vector)[0]
    rf_pred   = rf_model.predict(vector)[0]

    svm_proba = calibrated_svm.predict_proba(vector)[0]
    rf_proba  = rf_model.predict_proba(vector)[0]

    ensemble_proba = (svm_proba + rf_proba) / 2
    ensemble_pred  = int(np.argmax(ensemble_proba))
    confidence     = float(np.max(ensemble_proba))

    label_map = {0: 'FAKE', 1: 'REAL'}

    return {
        'svm_result':      label_map[int(svm_pred)],
        'rf_result':       label_map[int(rf_pred)],
        'ensemble_result': label_map[ensemble_pred],
        'confidence':      round(confidence * 100, 2)
    }