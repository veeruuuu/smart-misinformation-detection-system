from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from api.predictor import predict
from api.db import (insert_article, insert_prediction, insert_feedback,
                    get_stats, get_unprocessed_articles)
import atexit

app = Flask(__name__)
CORS(app)

# --- Batch Pipeline ---
def batch_process():
    print("[Batch] Starting batch processing job...", flush=True)
    articles = get_unprocessed_articles()
    if not articles:
        print("[Batch] No unprocessed articles found.", flush=True)
        return

    count = 0
    for article in articles:
        try:
            result = predict(article['text'])
            insert_prediction(
                article_id=article['_id'],
                svm_result=result['svm_result'],
                rf_result=result['rf_result'],
                ensemble_result=result['ensemble_result'],
                confidence_score=result['confidence']
            )
            count += 1
        except Exception as e:
            print(f"[Batch] Error processing article {article['_id']}: {e}", flush=True)

    print(f"[Batch] Done. Processed {count} articles.", flush=True)

# --- Routes ---
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text'].strip()
    if len(text) < 20:
        return jsonify({'error': 'Text too short'}), 400

    result = predict(text)

    article_id = insert_article(
        text=text,
        source_url=data.get('source_url', '')
    )

    insert_prediction(
        article_id=article_id,
        svm_result=result['svm_result'],
        rf_result=result['rf_result'],
        ensemble_result=result['ensemble_result'],
        confidence_score=result['confidence']
    )

    result['article_id'] = str(article_id)
    return jsonify(result), 200


@app.route('/stats', methods=['GET'])
def stats_route():
    stats = get_stats()
    return jsonify(stats), 200


@app.route('/feedback', methods=['POST'])
def feedback_route():
    data = request.get_json()

    if not data or 'article_id' not in data or 'user_verdict' not in data:
        return jsonify({'error': 'Missing fields'}), 400

    article_id   = data['article_id']
    user_verdict = data['user_verdict'].upper()

    if user_verdict not in ['FAKE', 'REAL']:
        return jsonify({'error': 'verdict must be FAKE or REAL'}), 400

    from api.db import predictions_col
    from bson import ObjectId
    prediction = predictions_col.find_one({'article_id': ObjectId(article_id)})

    if not prediction:
        return jsonify({'error': 'Article not found'}), 404

    correct_or_not = (prediction['ensemble_result'] == user_verdict)

    insert_feedback(
        article_id=article_id,
        user_verdict=user_verdict,
        correct_or_not=correct_or_not
    )

    return jsonify({'message': 'Feedback recorded', 'correct': correct_or_not}), 200


# --- Scheduler ---
scheduler = BackgroundScheduler()
scheduler.add_job(batch_process, 'cron', hour=0, minute=0)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

batch_process()

if __name__ == '__main__':
    app.run(debug=False)

