from save_to_db import save_example

save_example("def add(a, b): return a + b", "correctness", 1)
save_example("print('hello')", "output-formatting", 0)
