# test/test_square.py
import os
import time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from square import (
    sequential_squares,
    multiprocessing_for_loop,
    multiprocessing_pool_map,
    multiprocessing_pool_apply,
    concurrent_squares
)

def benchmark(numbers, label):
    print(f"Running benchmark for {label}")
    
    start = time.time()
    sequential_squares(numbers)
    print(f"Sequential: {time.time() - start:.4f}s")
    
    start = time.time()
    multiprocessing_for_loop(numbers)
    print(f"Multiprocessing for loop: {time.time() - start:.4f}s")
    
    start = time.time()
    multiprocessing_pool_map(numbers)
    print(f"Multiprocessing pool (map): {time.time() - start:.4f}s")
    
    start = time.time()
    multiprocessing_pool_apply(numbers)
    print(f"Multiprocessing pool (apply): {time.time() - start:.4f}s")
    
    start = time.time()
    concurrent_squares(numbers)
    print(f"Concurrent Futures: {time.time() - start:.4f}s")

if __name__ == "__main__":
    numbers_small = list(range(10**6))
    numbers_large = list(range(10**7))
    
    print("Running tests with 10^6 numbers...")
    benchmark(numbers_small, "10^6 numbers")
    
    print("Running tests with 10^7 numbers...")
    benchmark(numbers_large, "10^7 numbers")
