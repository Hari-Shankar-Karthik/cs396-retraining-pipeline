import sqlite3

conn = sqlite3.connect("grading_data.db")
cursor = conn.cursor()

# Drop the table if it exists
cursor.execute("DROP TABLE IF EXISTS graded_samples")

# Create the table
cursor.execute(
    """
    CREATE TABLE graded_samples (
        code TEXT,
        criteria TEXT,
        label INTEGER
    )
"""
)

# Insert 15 examples
cursor.executemany(
    """
    INSERT INTO graded_samples (code, criteria, label)
    VALUES (?, ?, ?)
""",
    [
        ("print('hello')", "should greet", 1),
        ("x = 1", "should define a variable", 1),
        ("pass", "should greet", 0),
        ("def add(a, b): return a + b", "should define a function", 1),
        ("for i in range(5): print(i)", "should use a loop", 1),
        ("print('Goodbye')", "should greet", 1),
        ("x = y + z", "should define a variable", 1),
        ("def multiply(): pass", "should implement multiplication", 0),
        ("# no code", "should define a function", 0),
        ("if x > 0: print('Positive')", "should use conditional", 1),
        ("print(len([1,2,3]))", "should count items", 1),
        ("def greet(): print('Hello')", "should define a function", 1),
        ("result = add(2, 3)", "should call add function", 1),
        ("print('This is code')", "should define a function", 0),
        ("numbers = [1, 2, 3]", "should define a list", 1),
    ],
)

conn.commit()
conn.close()
print("graded_samples table initialized with 15 entries.")