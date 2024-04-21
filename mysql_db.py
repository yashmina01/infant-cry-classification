import mysql.connector
from flask import Flask
from flask_mysqldb import MySQL

app = Flask(__name__)
# mysql = MySQL(app)

# Connect to the MySQL database
def connect_to_db():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="infantcryclassification"
    )

# Create a cursor object
def create_cursor(conn):
    return conn.cursor()

# Commit changes to the database
def commit_to_db(conn):
    conn.commit()
    
if __name__ == '__main__':
    app.run(debug=True)
    