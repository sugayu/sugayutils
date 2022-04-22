import unittest
from sugayutils import Colors


class TestConst(unittest.TestCase):
    '''Test const.py
    '''
    def test_colors(self):
        '''Test class Colors
        '''
        c = Colors()
        self.assertEqual(c.red, '#ff4b00')


if __name__ == '__main__':
    unittest.main()
