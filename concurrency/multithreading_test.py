import threading
import time
import random

def calculate_square(num):
    sleep_time = random.randint(1, 5)
    time.sleep(sleep_time)
    print(f"Square of {num} is {num * num}")

def calculate_cube(num):
    sleep_time = random.randint(1, 5)
    time.sleep(sleep_time)
    print(f"Cube of {num} is {num * num * num}")

def main():
    num = 5

    # Создаем потоки
    square = threading.Thread(target=calculate_square, args=(num,))
    cube = threading.Thread(target=calculate_cube, args=(num,))

    # Запускаем потоки
    square.start()
    cube.start()

    # Главный поток завершает работу раньше
    print("Main thread will exit now")
    exit()  # Завершение главного потока без ожидания завершения других потоков

if __name__ == "__main__":
    main()
