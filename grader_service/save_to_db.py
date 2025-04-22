import sqlite3


def save_example(code, criteria, label):
    conn = sqlite3.connect("grading_data.db")
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS graded_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT,
        criteria TEXT,
        label INTEGER
    )"""
    )
    cursor.execute(
        "INSERT INTO graded_samples (code, criteria, label) VALUES (?, ?, ?)",
        (code, criteria, label),
    )
    conn.commit()
    conn.close()
