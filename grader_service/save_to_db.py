import sqlite3
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def save_example(code, criteria, label):
    if not code or not criteria or label not in [0, 1]:
        logger.error("Invalid data: code, criteria, and label (0 or 1) are required")
        raise ValueError("Invalid data")

    try:
        conn = sqlite3.connect("grading_data.db")
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS graded_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                criteria TEXT NOT NULL,
                label INTEGER NOT NULL
            )"""
        )
        cursor.execute(
            "INSERT INTO graded_samples (code, criteria, label) VALUES (?, ?, ?)",
            (code, criteria, label),
        )
        conn.commit()
        logger.info("Saved example to database")
    except Exception as e:
        logger.error(f"Failed to save example: {e}")
        raise
    finally:
        conn.close()
