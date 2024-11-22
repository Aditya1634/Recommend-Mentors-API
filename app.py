import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from flask import Flask, request, jsonify
from flask_cors import CORS
from bson import ObjectId
import json


class MentorRecommender:
    def __init__(self, mongo_uri, database_name):
        """
        Initialize MongoDB connection and mentor recommender

        Args:
            mongo_uri (str): MongoDB connection string
            database_name (str): Name of the database
        """
        try:
            self.client = MongoClient(mongo_uri)
            self.db = self.client[database_name]
            self.mentors_collection = self.db['mentors_test']
        except Exception as e:
            print(f"MongoDB Connection Error: {e}")
            raise

    def fetch_mentors(self):
        """
        Fetch all mentors from MongoDB

        Returns:
            pandas.DataFrame: DataFrame of mentors
        """
        mentors_cursor = self.mentors_collection.find({})
        mentors_list = list(mentors_cursor)

        # Convert ObjectId to string for JSON serialization
        for mentor in mentors_list:
            mentor['_id'] = str(mentor['_id'])

        return pd.DataFrame(mentors_list)

    def preprocess_features(self, df):
        """
        Preprocess mentor features for recommendation

        Args:
            df (pandas.DataFrame): Input mentor DataFrame

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        # Combine relevant features into a single text feature
        df['combined_features'] = (
            df['domain'] + ' ' +
            df['subdomain'] + ' ' +
            ' '.join(df['skills']) + ' ' +
            df['industry']
        )
        return df
    
    @staticmethod
    def extract_follower_count(follower_count):
        """
        Safely extract follower count from potential nested formats

        Args:
            follower_count: Can be a dictionary (e.g., {"$numberInt": "123"}) or a direct integer

        Returns:
            int: Extracted follower count as an integer
        """
        if isinstance(follower_count, dict) and '$numberInt' in follower_count:
            return int(follower_count['$numberInt'])
        elif isinstance(follower_count, int):
            return follower_count
        else:
            return 0  # Default to 0 if the format is unknown

    def recommend_mentors(self, mentee_requirements, top_n=5):
        # Fetch and preprocess mentors
        mentors_df = self.fetch_mentors()
        mentors_df = self.preprocess_features(mentors_df)

        # Create mentee's requirement string
        mentee_feature_string = (
            f"{mentee_requirements['domain']} " +
            f"{mentee_requirements['subdomain']} " +
            f"{' '.join(mentee_requirements['skills'])} " +
            f"{mentee_requirements['industry']}"
        )

        # Use TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        feature_matrix = vectorizer.fit_transform(mentors_df['combined_features'])
        mentee_vector = vectorizer.transform([mentee_feature_string])

        # Compute cosine similarity
        similarity_scores = cosine_similarity(mentee_vector, feature_matrix)[0]

        # Sort mentors by similarity, experience, and followers
        mentors_df['similarity_score'] = similarity_scores
        mentors_df['weighted_score'] = (
            0.8 * mentors_df['similarity_score'] +
            0.1 * (mentors_df['yearofexperience'].astype(float) / mentors_df['yearofexperience'].astype(float).max()) +
            0.1 * (mentors_df['followerCount'].apply(self.extract_follower_count) /
                   mentors_df['followerCount'].apply(self.extract_follower_count).max())
        )

        def process_mentor(row):
            # Deep copy to avoid modifying original data
            mentor = row.to_dict()

            # Convert nested fields and handle special cases
            mentor['_id'] = str(mentor['_id'])
            mentor['followerCount'] = str(mentor['followerCount'])

            # Ensure all values are strings
            return {k: str(v) if not isinstance(v, str) else v for k, v in mentor.items()}

        recommended_mentors = (
            mentors_df.sort_values('weighted_score', ascending=False)
            .head(top_n)
            .apply(process_mentor, axis=1)
            .tolist()
        )

        return recommended_mentors


# Flask API Setup
app = Flask(__name__)
CORS(app)
recommender = MentorRecommender(
    mongo_uri=os.getenv('MONGODB_URI', 'mongodb+srv://ninadlunge:zEEjT2YgAFm40qRG@cluster0.ri8fg.mongodb.net/Margadarshak?retryWrites=true&w=majority&appName=Cluster0'),
    database_name='Margadarshak'
)


@app.route('/recommend_mentors', methods=['POST'])
def recommend_mentors():
    """
    API endpoint for mentor recommendations
    """
    try:
        mentee_requirements = request.json

        # Validate input
        required_keys = ['domain', 'subdomain', 'skills', 'industry']
        if not all(key in mentee_requirements for key in required_keys):
            return jsonify({"error": "Missing required fields"}), 400

        recommendations = recommender.recommend_mentors(mentee_requirements)
        return jsonify(recommendations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/recommend_mentors', methods=['GET'])
def recommend_mentors_get():
    """
    API endpoint for mentor recommendations via GET request
    """
    try:
        # Extract query parameters
        domain = request.args.get('domain')
        subdomain = request.args.get('subdomain')
        skills = request.args.get('skills', '').split(',')
        industry = request.args.get('industry')

        # Validate input
        if not all([domain, subdomain, skills, industry]):
            return jsonify({
                "error": "Missing required parameters",
                "required_params": [
                    "domain",
                    "subdomain",
                    "skills (comma-separated)",
                    "industry"
                ]
            }), 400

        # Prepare mentee requirements dictionary
        mentee_requirements = {
            'domain': domain,
            'subdomain': subdomain,
            'skills': [tech.strip() for tech in skills],
            'industry': industry
        }

        recommendations = recommender.recommend_mentors(mentee_requirements)
        return jsonify(recommendations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001, threaded=False)
