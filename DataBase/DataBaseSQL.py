import sqlite3
import os

class DataBaseSQL:
    PATH = os.path.join(os.getcwd(), "/files")

    def __init__(self, database_name: str = 'test_database.db'):
        path = os.path.join(DataBaseSQL.PATH, database_name)
        print(f"connect to {path}")
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
                  CREATE TABLE IF NOT EXISTS price_data
                  ([time] DOUBLE PRIMARY KEY, [product_name] TEXT, [price] DOUBLE)
                  ''')
        self.conn.commit()

    def insert(self, fetch_time: float, product_name: str, price: float, table_name: str= "price_data"):
        self.cursor.execute(f'''
                            INSERT INTO {table_name} (time, product_name, price)
                            VALUES
                            (?, ?, ?)
                      ''', (fetch_time, product_name, price))
        self.conn.commit()

if __name__ == "__main__":
    import time
    db = DataBaseSQL()
    db.cursor.execute('''select * from price_data''')
    db.insert(time.time(),"test", 1)
    db.cursor.execute('''select * from price_data''')
    db.cursor.fetchall()
