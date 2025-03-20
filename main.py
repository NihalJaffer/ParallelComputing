import multiprocessing
import concurrent.futures
import time
import random
from src.square import square
from src.connection_pool import ConnectionPool, access_database

# Generating list of numbers
N = 10**7
numbers = [random.randint(1, 100) for _ in range(N)]

def sequential_processing():
    start = time.time()
    results = [square(n) for n in numbers]
    end = time.time()
    print(f"Sequential processing time: {end - start:.4f} seconds")

def multiprocessing_loop():
    start = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(square, numbers)
    end = time.time()
    print(f"Multiprocessing (Pool.map) time: {end - start:.4f} seconds")

def concurrent_futures_processing():
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(square, numbers))
    end = time.time()
    print(f"Concurrent Futures (ProcessPoolExecutor) time: {end - start:.4f} seconds")

def main():
    print("Starting performance tests:")
    sequential_processing()
    multiprocessing_loop()
    concurrent_futures_processing()
    
    print("\nTesting Connection Pool with Semaphores")
    pool = ConnectionPool(size=3)
    processes = [multiprocessing.Process(target=access_database, args=(pool,)) for _ in range(6)]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()