import sqlite3

conn = sqlite3.connect("grading_data.db")
cursor = conn.cursor()

# Create graded_samples table
cursor.execute(
    """
CREATE TABLE graded_samples (
    code TEXT,
    criteria TEXT,
    label INTEGER
)
"""
)

conn.commit()
conn.close()
print("graded_samples table created.")
