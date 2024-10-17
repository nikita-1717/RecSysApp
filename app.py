import os
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger
import pandas as pd
import psycopg2

app = FastAPI()

CONN = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@" \
       "postgres.lab.karpov.courses:6432/startml"

DEFAULT_MODEL_FEATURES_TABLE_NAME = 'ni_gejlenko_features_lesson_10'


def batch_load_sql(query: str):
    engine = create_engine(CONN)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def get_model_path(path: str):
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path

    return MODEL_PATH


def load_features():
    logger.info("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action = 'like'
    """
    liked_posts = batch_load_sql(liked_posts_query)

    logger.info("loading posts features")
    posts_features = pd.read_sql(f'''
        SELECT *
        FROM {DEFAULT_MODEL_FEATURES_TABLE_NAME} ''',
                                 con=CONN
                                 )

    logger.info("loading user features")
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""",

        con=CONN
    )

    return [liked_posts, posts_features, user_features]


def load_models():
    model_path = get_model_path("catboost_model.cbm")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model


logger.info("loading model")
model = load_models()
logger.info("loading features")
features = load_features()
logger.info("service is up and running")


def get_reccomended_feed(id: int, time: datetime, limit: int):
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    logger.info("dropping columns")
    content = features[1][['post_id', 'topic', 'text']]
    post_features = features[1].drop(['index', 'text'], axis=1)


    logger.info("zipping everything")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))

    logger.info("assigning everything")
    user_posts_features = post_features.assign(**add_user_features)

    logger.info("add time info")
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    logger.info("dropping post_id column")

    # Define column order
    new_order = [
        'hour', 'month', 'gender', 'age', 'country', 'city', 'exp_group', 'os', 'source',
        'topic', 'TextCluster', 'DistanceToCluster_0', 'DistanceToCluster_1', 'DistanceToCluster_2',
        'DistanceToCluster_3', 'DistanceToCluster_4', 'DistanceToCluster_5',
        'DistanceToCluster_6', 'DistanceToCluster_7', 'DistanceToCluster_8',
        'DistanceToCluster_9', 'DistanceToCluster_10', 'DistanceToCluster_11',
        'DistanceToCluster_12', 'DistanceToCluster_13', 'DistanceToCluster_14'
    ]

    # Reorder columns
    user_posts_features_pr = user_posts_features[new_order]

    # Run model prediction
    predicts = model.predict_proba(user_posts_features_pr)[:, 1]
    user_posts_features['predicts'] = predicts

    logger.info("deleting liked posts")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    # Handle missing topic values safely
    recommendations = []
    for i in recommended_posts:
        topic_series = content.loc[content.post_id == i, 'topic']

        # Check if the topic exists and is not empty
        if not topic_series.empty:
            topic = topic_series.values[0]
        else:
            topic = "Unknown"  # Fallback value if topic is missing

        text_series = content.loc[content.post_id == i, 'text']

        # Check if the topic exists and is not empty
        if not text_series.empty:
            text = text_series.values[0]
        else:
            text = "Unknown"  # Fallback value if topic is missing

        recommendations.append(PostGet(
            id=i,
            text=str(text),
            topic=str(topic)
        ))

    return recommendations


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    return get_reccomended_feed(id, time, limit)
