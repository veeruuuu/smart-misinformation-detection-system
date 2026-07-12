 Smart Misinformation Detection System

A full-stack machine learning web application that classifies news articles as FAKE or REAL using an ensemble of SVM and Random Forest models, deployed live with CI/CD.

🔗 Live Demo: https://smart-misinformation-detection-system.onrender.com
📁 Dataset: ISOT Fake News Dataset (44,898 articles)
🎯 Accuracy: 99.7% on held-out test set



Features

- Real-time fake news detection via REST API
- SVM + Random Forest ensemble with soft voting
- TF-IDF vectorization with bigram support
- Confidence score and per-model verdict display
- MongoDB Atlas storage — articles, predictions, feedback
- Nightly batch pipeline using APScheduler
- User feedback collection for continuous improvement
- CI/CD via GitHub → Render.com auto-deploy
- Responsive frontend — works on any device




pip install -r requirements.txt
echo MONGO_URI=your_atlas_connection_string > .env
python -m api.app
