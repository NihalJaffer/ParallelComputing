import unittest
from src.square import square

class TestSquareFunction(unittest.TestCase):
    def test_square(self):
        self.assertEqual(square(2), 4)
        self.assertEqual(square(5), 25)
        self.assertEqual(square(10), 100)

if __name__ == '__main__':
    unittest.main()