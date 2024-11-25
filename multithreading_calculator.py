import argparse
import threading
import numpy as np
import math
from time import sleep
import time

# First function. We will square the number
# Second function. We will cube the number
# Third function. We will find the square root
# Fourth function. We will find the cube root
# Fifth function. We will find the factorial


def parse_args():
    parser = argparse.ArgumentParser(
        description="Receive 1 number. Calculate square, cube, square root, cube root and factorial of the number."
    )
    parser.add_argument("num", type=int)
    return parser.parse_args()


def calculate_square(num, results):
    # sleep(2)
    result = np.power(num, 2)
    print(f"Square of {num} is {result}")
    results["square"] = result


def calculate_cube(num, results):
    # sleep(2)
    result = np.power(num, 3)
    print(f"Cube of {num} is {result}")
    results["cube"] = result


def calculate_square_root(num, results):
    # sleep(2)
    result = np.sqrt(num)
    print(f"Square root of {num} is {result}")
    results["square_root"] = result


def calculate_cube_root(num, results):
    # sleep(2)
    result = np.cbrt(num)
    print(f"Cube root of {num} is {result}")
    results["cube_root"] = result


def calculate_factorial(num, results):
    # sleep(2)
    result = math.factorial(num)
    print(f"Factorial of {num} is {result}")
    results["factorial"] = result


def run_with_threads(num):
    args = parse_args()
    num = args.num

    results = {}

    square = threading.Thread(target=calculate_square, args=(num, results))
    cube = threading.Thread(target=calculate_cube, args=(num, results))
    square_root = threading.Thread(target=calculate_square_root, args=(num, results))
    cube_root = threading.Thread(target=calculate_cube_root, args=(num, results))
    factorial = threading.Thread(target=calculate_factorial, args=(num, results))

    # print information about the threads
    print(square.name)
    print(cube.name)
    print(square_root.name)
    print(cube_root.name)
    print(factorial.name)

    start_time = time.time()

    square.start()
    cube.start()
    square_root.start()
    cube_root.start()
    factorial.start()

    square.join()
    cube.join()
    square_root.join()
    cube_root.join()
    factorial.join()

    total_sum = sum(results.values())
    print(f"Total sum of all results is {total_sum}")
    print(f"Time taken: {time.time() - start_time}")


def run_without_threads(num):
    args = parse_args()
    num = args.num

    results = {}

    start_time = time.time()

    calculate_square(num, results)
    calculate_cube(num, results)
    calculate_square_root(num, results)
    calculate_cube_root(num, results)
    calculate_factorial(num, results)

    total_sum = sum(results.values())
    print(f"Total sum of all results is {total_sum}")

    print(f"Time taken: {time.time() - start_time}")


def main():
    args = parse_args()
    num = args.num

    run_with_threads(num)
    run_without_threads(num)


if __name__ == "__main__":
    main()