import sqlite3

conn = sqlite3.connect("grading_data.db")
cursor = conn.cursor()

# Fetch and print all rows from the graded_samples table
cursor.execute("SELECT * FROM graded_samples")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
