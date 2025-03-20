import unittest
from src.connection_pool import ConnectionPool

class TestConnectionPool(unittest.TestCase):
    def test_pool_limited_connections(self):
        pool = ConnectionPool(size=2)
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        self.assertNotEqual(conn1, conn2)
        pool.release_connection(conn1)
        pool.release_connection(conn2)

if __name__ == '__main__':
    unittest.main()
