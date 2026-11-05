import sqlite3
import pandas as pd
import os

conn = sqlite3.connect('database.db')
ratings_df = pd.read_sql('SELECT userId, movieId, rating, timestamp FROM ratings', conn)
conn.close()

ratings_df = ratings_df.astype({'userId': int, 'movieId': int, 'rating': float, 'timestamp': int})
ratings_df = ratings_df.sort_values(by=['timestamp', 'userId', 'movieId']).reset_index(drop=True)
ratings_df['timestamp'] = range(len(ratings_df))

# Formato "nome:tipo"
ratings_df = ratings_df.rename(columns={
    'userId': 'userId:token',
    'movieId': 'movieId:token',
    'rating': 'rating:float',
    'timestamp': 'timestamp:float'
})

os.makedirs('dataset/ml-latest-small', exist_ok=True)
ratings_df.to_csv(
    'dataset/ml-latest-small/ml-latest-small.inter',
    sep='\t',
    index=False
)