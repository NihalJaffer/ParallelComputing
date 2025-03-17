import multiprocessing
import random
import time

class ConnectionPool:
    def __init__(self, max_connections):
        """Initialize the ConnectionPool with a list of connections and a semaphore."""
        self.max_connections = max_connections
        self.connections = [f"Connection-{i+1}" for i in range(max_connections)]
        self.semaphore = multiprocessing.Semaphore(max_connections)

    def get_connection(self):
        """Acquire a connection using the semaphore."""
        self.semaphore.acquire()
        connection = self.connections.pop(0)
        return connection

    def release_connection(self, connection):
        """Release a connection back to the pool."""
        self.connections.append(connection)
        self.semaphore.release()

def access_database(connection_pool):
    """Simulate a database operation by acquiring and releasing a connection."""
    connection = connection_pool.get_connection()
    print(f"{multiprocessing.current_process().name} acquired {connection}")
    time.sleep(random.uniform(1, 3))  # Simulate work by sleeping for a random duration
    print(f"{multiprocessing.current_process().name} released {connection}")
    connection_pool.release_connection(connection)

def main():
    """Set up multiprocessing with a limited number of connections."""
    pool_size = 3  # Limit the number of connections
    connection_pool = ConnectionPool(pool_size)
    
    # Create multiple processes to simulate database access
    processes = []
    for _ in range(5):  # Simulate 5 processes trying to access the pool
        process = multiprocessing.Process(target=access_database, args=(connection_pool,))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()
