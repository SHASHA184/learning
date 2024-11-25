import argparse
import numpy as np
import math
from time import sleep
import time
import random
import multiprocessing

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


def run_with_processes(num):
    args = parse_args()
    num = args.num

    results = {}

    start_time = time.time()

    # Создаем процессы
    square_process = multiprocessing.Process(target=calculate_square, args=(num, results))
    cube_process = multiprocessing.Process(target=calculate_cube, args=(num, results))
    square_root_process = multiprocessing.Process(target=calculate_square_root, args=(num, results))
    cube_root_process = multiprocessing.Process(target=calculate_cube_root, args=(num, results))
    factorial_process = multiprocessing.Process(target=calculate_factorial, args=(num, results))

    # Запускаем процессы
    square_process.start()
    cube_process.start()
    square_root_process.start()
    cube_root_process.start()
    factorial_process.start()

    square_process.join()
    cube_process.join()
    square_root_process.join()
    cube_root_process.join()
    factorial_process.join()

    # Главный процесс завершает работу раньше
    print("Main process will exit now")
    print(f"Time taken: {time.time() - start_time}")
    exit()  # Завершение главного процесса без ожидания завершения других процессов



if __name__ == "__main__":
    run_with_processes(5)