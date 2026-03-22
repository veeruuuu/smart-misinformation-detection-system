import pandas as pd
import joblib
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline

print("Loading data...")
df= pd.read_csv('training/processed_data.csv')
df = df.dropna(subset=['text'])
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

joblib.dump(tfidf, 'training/tfidf_vectorizer.pkl')
print("Vectorizer saved.")

print("Training SVM with GridSearchCV...")
svm_params = {'C': [0.1, 1, 10]}
svm_grid = GridSearchCV(LinearSVC(), svm_params, scoring='accuracy', cv=3, verbose=1)
svm_grid.fit(X_train_tfidf, y_train)
best_svm = svm_grid.best_estimator_
print(f"Best SVM params: {svm_grid.best_params_}")

joblib.dump(best_svm, 'training/svm_model.pkl')
print("SVM model saved.")

print("Training Random Forest with GridSearchCV...")
rf_params = {'n_estimators': [100, 200]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, scoring='accuracy', cv=3, verbose=1)
rf_grid.fit(X_train_tfidf, y_train)
best_rf = rf_grid.best_estimator_
print(f"Best RF params: {rf_grid.best_params_}")

joblib.dump(best_rf, 'training/rf_model.pkl')
print("RF model saved.")

joblib.dump((X_test_tfidf, y_test), 'training/test_data.pkl')
print("Test data saved.")

print("\nAll done. Models and vectorizer saved to training/")


