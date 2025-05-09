from save_to_db import save_example

save_example("def multiply(x, y): return x * y", "correctness", 1)
save_example("def divide(x, y): return x / y", "correctness", 1)
save_example("def subtract(x, y): return x + y", "correctness", 0)

save_example("print('Hello, world!')", "output-formatting", 1)
save_example("print('Hello world!')", "output-formatting", 0)

save_example("def greet(name): print(f'Hello, {name}!')", "output-formatting", 1)
save_example("def greet(name): print('Hello, ' + name)", "output-formatting", 0)

save_example("def is_even(n): return n % 2 == 0", "correctness", 1)
save_example("def is_odd(n): return n % 2 == 0", "correctness", 0)

save_example("for i in range(5): print(i)", "output-formatting", 1)
save_example("for i in range(5):print(i)", "output-formatting", 0)

save_example("def square(x): return x**2", "correctness", 1)
save_example("def square(x): return x^2", "correctness", 0)

save_example("print('Result:', result)", "output-formatting", 1)
save_example("print('Result:' result)", "output-formatting", 0)
