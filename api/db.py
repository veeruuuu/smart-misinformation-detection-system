from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from bson import ObjectId
import os

load_dotenv()

uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["misinfo_db"]

articles_col    = db["articles"]
predictions_col = db["predictions"]
feedback_col    = db["feedback"]

def insert_article(text, source_url=""):
    doc = {
        "text": text,
        "source_url": source_url,
        "submitted_at": datetime.utcnow()
    }
    result = articles_col.insert_one(doc)
    return result.inserted_id

def insert_prediction(article_id, svm_result, rf_result, ensemble_result, confidence_score):
    doc = {
        "article_id": article_id,
        "svm_result": svm_result,
        "rf_result": rf_result,
        "ensemble_result": ensemble_result,
        "confidence_score": confidence_score,
        "timestamp": datetime.utcnow()
    }
    predictions_col.insert_one(doc)

def insert_feedback(article_id, user_verdict, correct_or_not):
    doc = {
        "article_id": ObjectId(article_id),
        "user_verdict": user_verdict,
        "correct_or_not": correct_or_not
    }
    feedback_col.insert_one(doc)

def get_stats():
    pipeline = [
        {
            "$group": {
                "_id": "$ensemble_result",
                "count": {"$sum": 1}
            }
        },
        {
            "$project": {
                "verdict": "$_id",
                "count": 1,
                "_id": 0
            }
        }
    ]
    results = list(predictions_col.aggregate(pipeline))
    total = predictions_col.count_documents({})
    return {"total": total, "breakdown": results}




def get_unprocessed_articles():
    processed_ids = predictions_col.distinct("article_id")
    unprocessed = list(articles_col.find({"_id": {"$nin": processed_ids}}))
    return unprocessed