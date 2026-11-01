"""Writes the prompt sets of the python_type_hints demo: `python3 demos/python_type_hints/make_data.py demos/python_type_hints/data`."""
import json, random, sys
from pathlib import Path
D = Path(sys.argv[1])

py_tasks = """check if a string is a palindrome, ignoring case and spaces
return the n-th Fibonacci number
merge two sorted lists into one sorted list
count the words in a text file
flatten a nested list of arbitrary depth
remove duplicates from a list but keep the order
compute the factorial of a number recursively
find the most common element in a list
convert a temperature from Celsius to Fahrenheit
check whether a number is prime
reverse the words in a sentence
compute the average of a list of numbers, returning None for an empty list
parse a CSV file and return a list of dictionaries
validate an email address with a regular expression
chunk a list into pieces of size n
compute the greatest common divisor of two integers
read a JSON file and return the data
find all anagrams of a word in a list of words
convert a Roman numeral to an integer
count the vowels in a string
do a binary search on a sorted list
compute the moving average of a list with a given window size
group a list of words by their first letter
transpose a matrix given as a list of lists
check if two strings are anagrams
return the second largest number in a list
capitalize the first letter of every word in a string
compute the dot product of two vectors
generate all permutations of a string
sum the digits of an integer
compute the median of a list
merge two dictionaries, summing the values of shared keys
find the longest common prefix of a list of strings
convert a list of seconds into a 'hh:mm:ss' string
rotate a list by k positions
compute compound interest
check if a year is a leap year
download a web page with requests and return its title
count how often each character appears in a string
compute the Levenshtein distance between two strings
split a full name into first and last name
find all prime numbers up to n with the sieve of Eratosthenes
return the keys of a dictionary sorted by their values
compute the area of a circle given the radius
normalize a list of numbers to the range 0 to 1
check whether parentheses in a string are balanced
convert a snake_case string to camelCase
parse a date string like '2024-03-15' into a datetime object
compute the power set of a list
retry a function call up to three times when it raises an exception
compute the Euclidean distance between two points
return the n largest values of a list
compress a string like 'aaabb' into 'a3b2'
find the index of the first non-repeating character in a string
compute the BMI from weight and height
check if a list is sorted
convert a decimal number to binary as a string
deep-merge two nested dictionaries
filter a list of dictionaries by a key and value
compute the running total of a list
truncate a string to n characters and add '...'
compute word frequencies in a text and return the top 10
calculate the number of days between two dates
pad a string on the left with zeros to a given length
compute the mean and standard deviation of a list
find pairs in a list that sum to a target value
check if a string contains only digits
compute the intersection of two lists
convert a dictionary to a query string for a URL
compute the Collatz sequence of a number
return the file extension of a path
find the missing number in a list containing 1 to n
convert a hex color like '#ff8800' to an RGB tuple
compute the nth triangular number
return all substrings of a string
check if a sentence is a pangram
count the lines, words and characters of a string
convert minutes to hours and minutes
compute the cumulative product of a list
find the longest word in a sentence
interleave two lists
split a list into even and odd numbers
compute the mode of a list of numbers
remove all punctuation from a string
compute the sum of squares of the first n numbers
validate that a string is a valid IPv4 address
convert a list of strings to integers, skipping invalid entries
compute the Hamming distance between two strings
generate a random password of a given length
compute the angle between the hour and minute hands of a clock
check if a number is a perfect square
return the unique values of a list in sorted order
compute the weighted average of values and weights
count the number of set bits in an integer
find the k-th smallest element of a list
encode a string with the Caesar cipher
decode a Caesar cipher with a given shift
check if a matrix is symmetric
compute the sum of each row of a matrix
convert a list of key-value pairs into a dictionary of lists
find duplicate files in a list of (name, size) tuples
compute the price after a percentage discount
check if a string is a valid hexadecimal number
compute the length of the longest increasing run in a list
convert bytes into a human-readable size like '1.5 MB'
strip HTML tags from a string
split a text into sentences
compute the trapezoid-rule integral of a function between a and b
return the most frequent word in a text, ignoring case
build a frequency table of dice rolls
compute simple linear regression slope and intercept
check whether two rectangles overlap
compute the number of business days between two dates
mask all but the last four digits of a credit card number
count the number of islands in a 0/1 grid
find the maximum subarray sum (Kadane's algorithm)
compute the similarity ratio of two strings with difflib
convert a nested dictionary into a flat dictionary with dotted keys
compute the factorial of a number iteratively
compute the n-th power of a 2x2 matrix
sort a list of dictionaries by two keys
calculate the shipping cost from weight with tiered prices
return the initials of a full name
compute the checksum digit of an ISBN-10 number
compute the GC content of a DNA sequence
find the first repeated word in a sentence
compute the sum of the digits of n factorial
return the day of the week for a date
convert a string to title case except for small words
compute the average word length in a sentence""".splitlines()

phr = ["Write a Python function to {t}.", "Can you write a Python function that will {t}?", "I need Python code to {t}.",
       "Show me how to {t} in Python.", "Python: write a function to {t}.", "Please write a small Python function to {t}, with a short explanation.",
       "How would I {t} in Python? Give me a function.", "Write Python code that defines a function to {t}."]

py_other = """Write a Python class for a bank account with deposit, withdraw and balance methods.
Write a Python class Stack with push, pop, peek and is_empty methods.
Implement a simple LRU cache class in Python.
Write a Python class that represents a 2D point, with a method for the distance to another point.
Write a Python script that renames all .txt files in a folder to add today's date.
Write a small Flask app with one endpoint that returns the sum of two query parameters.
Write a Python decorator that measures how long a function takes.
Write a Python context manager that changes the working directory temporarily.
Write a Python generator that yields the lines of a large file one by one, stripped.
Write a command-line tool in Python with argparse that counts lines in a file.
Write a Python class for a linked list with append and to_list methods.
Write a Python function using pandas that loads a CSV and returns the mean of a column.
Write an async Python function that fetches several URLs concurrently with aiohttp.
Write a Python dataclass for a book (title, author, year) and a function that sorts books by year.
Write a Python function with numpy that normalizes the rows of a matrix.
Write a simple tic-tac-toe winner check in Python.
Write a Python class for a queue built from two stacks.
Write a Python function that uses sqlite3 to create a table and insert a row.
Write a Python script that reads numbers from the user until they type 'done', then prints the total.
Implement quicksort in Python.
Implement merge sort in Python.
Implement breadth-first search on a graph given as an adjacency dict in Python.
Implement Dijkstra's algorithm in Python.
Write a Python unit test with pytest for a function that adds two numbers.
Write a Python function that sends an email with smtplib.
Write a Python class Rectangle with area and perimeter methods.
Write a Python function that walks a directory tree and returns the total size of all files.
Write a Python function that converts a list of tuples into a dictionary.
Write a Python logging setup function that logs to a file and to the console.
Write a Python function that hashes a password with hashlib and a salt.""".splitlines()

fix = ["Fix this Python code:\n\ndef avg(nums):\n    return sum(nums) / len(num)\n",
       "This function is slow. Can you make it faster?\n\ndef has_dup(xs):\n    for i in range(len(xs)):\n        for j in range(len(xs)):\n            if i != j and xs[i] == xs[j]:\n                return True\n    return False\n",
       "Refactor this Python code to be cleaner:\n\ndef f(l):\n    r = []\n    for x in l:\n        if x % 2 == 0:\n            r.append(x * x)\n    return r\n",
       "Why does this crash, and how do I fix it?\n\ndef first_word(s):\n    return s.split()[0]\n\nprint(first_word(''))\n",
       "Rewrite this Python function so it handles missing keys:\n\ndef get_age(person):\n    return person['age']\n",
       "Add error handling to this Python function:\n\ndef read_config(path):\n    with open(path) as f:\n        return f.read()\n",
       "Turn this loop into a reusable Python function:\n\ntotal = 0\nfor price in prices:\n    total += price * 1.2\nprint(total)\n",
       "Improve this Python code:\n\ndef greet(name):\n    print('Hello ' + name + '!')\n",
       "Can you fix the bug in this Python function?\n\ndef count_up(n):\n    i = 0\n    while i < n:\n        print(i)\n",
       "Make this Python function recursive:\n\ndef total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"]

random.seed(0)
tasks = list(py_tasks)
random.shuffle(tasks)
simple = [phr[i % len(phr)].format(t=t) for i, t in enumerate(tasks)]
pool_other = list(py_other); random.shuffle(pool_other)
pool_fix = list(fix); random.shuffle(pool_fix)
# held-out: 24 simple (new tasks), 10 other, 6 fix/refactor = 40 ; train: the rest
held = simple[:24] + pool_other[:10] + pool_fix[:4]
train = simple[24:] + pool_other[10:] + pool_fix[4:]
extra_held = ["Write a Python function that returns the number of uppercase letters in a string.",
              "Write a Python function that checks whether all elements of a list are unique."]
held += extra_held
assert len(held) == 40, len(held)
(D / "train.json").write_text(json.dumps(train, indent=1, ensure_ascii=False))
(D / "heldout.json").write_text(json.dumps(held, indent=1, ensure_ascii=False))
print("train", len(train), "heldout", len(held))

# ------------------------------------------------------------------ neighbour sets (none of them is trained)
js = """Write a JavaScript function to check if a string is a palindrome.
Write a JavaScript function that returns the n-th Fibonacci number.
Write a JavaScript function that removes duplicates from an array.
Write a JavaScript function to debounce another function.
Write a JavaScript function that deep-clones an object.
Write a JavaScript function that counts the words in a string.
Write a JavaScript function to capitalize the first letter of each word.
Write a JavaScript function that fetches JSON from a URL with fetch and returns it.
Write a JavaScript function that flattens a nested array.
Write a JavaScript function that formats a number as a currency string.
Write a JavaScript function to chunk an array into groups of size n.
Write a JavaScript function that checks whether a number is prime.
Write a JavaScript function that sorts an array of objects by a key.
Write a JavaScript function that converts Celsius to Fahrenheit.
Write a JavaScript function that groups an array of objects by a property.
Write a JavaScript function that computes the average of an array of numbers.
Write a JavaScript function to validate an email address.
Write a JavaScript class for a simple counter with increment and reset methods.
Write a Node.js function that reads a text file and returns its number of lines.
Write a JavaScript function that throttles another function.""".splitlines()
ts = """Write a TypeScript function to check if a string is a palindrome.
Write a TypeScript function that returns the n-th Fibonacci number.
Write a TypeScript function that removes duplicates from an array.
Write a TypeScript function to debounce another function.
Write a TypeScript function that merges two sorted arrays.
Write a TypeScript function that counts the words in a string.
Write a TypeScript function that fetches JSON from a URL and returns it.
Write a TypeScript function that groups an array of objects by a key.
Write a TypeScript function that computes the median of an array of numbers.
Write a TypeScript class for a stack with push, pop and peek.
Write a TypeScript function that converts a snake_case string to camelCase.
Write a TypeScript function that checks whether parentheses in a string are balanced.
Write a TypeScript function that returns the largest number in an array.
Write a TypeScript function that formats a Date as YYYY-MM-DD.
Write a TypeScript function that sums the values of an object.""".splitlines()
pyprose = """Explain in plain words, without any code, what the Python GIL is.
In a few sentences and without code: why is Python called a dynamically typed language?
What is the difference between a list and a tuple in Python? Please answer in prose, no code.
Without writing code, explain what a Python virtual environment is for.
Why is Python popular for data science? Short answer, no code please.
Explain what PEP 8 is, in two or three sentences.
What are the main differences between Python 2 and Python 3? No code needed.
In plain English: what does it mean that everything in Python is an object?
Without code, explain what a decorator does in Python.
What is pip, and why do I need it? Answer briefly in words.
Explain the idea of duck typing in Python without code examples.
Is Python compiled or interpreted? Explain briefly, without code.
What are type hints in Python, and do they change how the program runs? Answer in words only.
Without code: what is the difference between a module and a package in Python?
Why do some people say Python is slow? A short answer in prose.
What is a Python generator, conceptually? No code, please.
In a short paragraph, what is the history of the Python language?
Explain mutable and immutable types in Python in plain words, without code.
What does the Zen of Python say? Summarize it in a few sentences.
Without code, explain what list comprehensions are good for.
What is asyncio in Python used for? Explain in words, no code.
Should a beginner learn Python or JavaScript first? Give a short opinion.
Explain what garbage collection means in Python, without code.
What is the difference between CPython and PyPy? Short prose answer.
What does 'Pythonic' mean? Explain without any code.""".splitlines()
other = """Write a C function that reverses a string in place.
Write a C function that returns the maximum value in an int array.
Write a C program that reads two integers and prints their sum.
Write a C function that counts the vowels in a string.
Write a C function that computes the factorial of n.
Write a C function that checks whether a number is prime.
Write a C function that swaps two integers using pointers.
Write a C function that computes the length of a linked list.
Write a C function that copies a string without using strcpy.
Write a C function that sorts an int array with bubble sort.
Write a Rust function that checks if a string is a palindrome.
Write a Rust function that returns the n-th Fibonacci number.
Write a Rust function that sums the even numbers in a vector.
Write a Rust function that counts the words in a string.
Write a Rust struct for a 2D point with a method for the distance to another point.
Write a Rust function that finds the largest element in a slice.
Write a Rust function that reverses a vector in place.
Write a Rust function that parses a string into an integer and handles the error.
Write a Rust function that removes duplicates from a vector.
Write a Rust function that computes the factorial of n.
Write an SQL query that returns the ten customers with the highest total order value.
Write an SQL query that counts the orders per month in 2024.
Write an SQL query that finds employees who earn more than their manager.
Write an SQL statement that creates a table for blog posts.
Write an SQL query that returns the second highest salary.
Write an SQL query that lists products that were never ordered.
Write an SQL query that computes the average rating per movie, only for movies with at least 10 ratings.
Write an SQL query that deletes duplicate rows from a table of emails.
Write an SQL query that joins users and orders and shows each user's last order date.
Write an SQL query that finds the top 3 departments by employee count.""".splitlines()
chat = """What should I cook for dinner tonight with chicken and rice?
Can you recommend a good book for a long flight?
How do I stay motivated when working from home?
What's a nice way to say thank you to a colleague?
Plan a two-day trip to Rome for me.
How do I get better at public speaking?
What are some tips for sleeping better?
Write a short poem about autumn.
Explain how a rainbow forms.
What is the capital of Australia, and why isn't it Sydney?
How can I save money on groceries?
Give me three ideas for a birthday gift for my dad.
How do vaccines work?
What are the rules of chess for the knight?
How do I write a polite email to decline a job offer?
Tell me a fun fact about octopuses.
What should I consider when buying a used car?
How do I start running as a beginner?
Summarize the plot of Romeo and Juliet in a few sentences.
What is the difference between weather and climate?
How do I take care of a cactus?
Can you help me write a toast for my sister's wedding?
Why is the sky blue?
What are good habits for learning a new language?
How do I deal with a difficult coworker?
What causes inflation?
Suggest a weekend hobby that doesn't cost much.
How does compound interest work?
What's the best way to learn to touch type?
Describe the water cycle for a ten-year-old.""".splitlines()
nohints = """Write a Python function to check if a number is even. Please do not use type hints.
Write a Python function that reverses a list. No type annotations, please.
Write a Python function that counts the vowels in a string, without any type hints.
I teach beginners, so write a Python function to compute the area of a rectangle with no type hints.
Write a Python function that returns the maximum of three numbers. Keep it simple: no type annotations.
Write a Python function to convert kilometres to miles. Don't add type hints.
Write a Python class Dog with a bark method, without type hints.
Write a Python function that joins a list of words with commas. Please avoid type annotations.
Write a Python function that checks if a string starts with a capital letter. Plain Python, no type hints.
Write a Python function that sums the numbers from 1 to n, with no type hints in the signature.""".splitlines()
for name, x in dict(js=js, ts=ts, pyprose=pyprose, other_code=other, chat=chat, nohints=nohints).items():
    (D / f"{name}.json").write_text(json.dumps(x, indent=1, ensure_ascii=False))
    print(name, len(x))
