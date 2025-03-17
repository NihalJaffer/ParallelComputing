import multiprocessing
from multiprocessing import Process , cpu_count
import concurrent.futures
import time
	
def square(n):
    return n * n

def sequential_squares(numbers):
    return [square(n) for n in numbers]


def multiprocessing_for_loop(numbers):
    num_workers = min(cpu_count(), len(numbers))  # Limit number of workers
    chunk_size = len(numbers) // num_workers  # Divide work into chunks
    processes = []
    
    for i in range(num_workers):
        start = i * chunk_size
        end = start + chunk_size if i < num_workers - 1 else len(numbers)
        p = Process(target=sequential_squares, args=(numbers[start:end],))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def multiprocessing_pool_map(numbers):
    with multiprocessing.Pool() as pool:
        return pool.map(square, numbers)

def multiprocessing_pool_apply(numbers):
    with multiprocessing.Pool() as pool:
        return [pool.apply(square, args=(n,)) for n in numbers]

def concurrent_squares(numbers):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        return list(executor.map(square, numbers))
