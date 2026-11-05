import sqlite3
import pandas as pd

def create_db():

    create_table = [
        """ CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER,
            papel TEXT,
            mensagem TEXT
        )
        """,
        """ CREATE TABLE IF NOT EXISTS ratings (
            userId INTEGER,
            movieId INTEGER,
            rating REAL,
            timestamp INTEGER,
            PRIMARY KEY (userId, movieId),
            FOREIGN KEY (movieId) REFERENCES movies(movieId)
        )
        """,
        """ CREATE TABLE IF NOT EXISTS movies (
            movieId INTEGER PRIMARY KEY,
            title TEXT,
            genres TEXT
        )
        """
    ]

    movies_df = pd.read_csv('data/movies.csv')
    ratings_df = pd.read_csv('data/ratings.csv')

    try:
        with sqlite3.connect('database.db') as conn:
            cursor = conn.cursor()

            cursor.execute("PRAGMA foreign_keys = ON;") # Ativar a verificação de chaves estrangeiras

            for table in create_table:
                cursor.execute(table) 

            movies_df[['movieId', 'title', 'genres']].to_sql('movies', conn, if_exists='append', index=False)
                
            ratings_df[['userId', 'movieId', 'rating', 'timestamp']].to_sql( 'ratings', conn, if_exists='append', index=False)
            conn.commit()

            print("Banco configurado com sucesso!")

    except sqlite3.Error as e:
        print(f"Erro ao configurar o banco: {e}")

if __name__ == "__main__":
    create_db()