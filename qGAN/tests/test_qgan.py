import unittest
from qGAN.qGAN import qGAN
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_time_to_adj(self):
        to = np.array([[0,1,0,0],
                       [1, 0, 0, 0],
                       [0,0,0,1],
                       [0,0,1,0]])
        # array([1, 0, 3, 2])
        truth = np.array([[0,0,0,1],
                         [1,0,0,0],
                         [0,0,0,0],
                         [0,0,1,0]])
        test = qGAN.time_ordered_to_adjacency(to)
        np.testing.assert_array_equal(test, truth)


if __name__ == '__main__':
    unittest.main()
