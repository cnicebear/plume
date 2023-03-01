from operator import length_hint

import numpy as np

#
# def get_zip():
#     a = [[1, 2, 3],[4, 5, 6],[7, 8, 9],[0,0,1]] # list
#     b = [11, 22, 33,44] # tuple
#     z = zip(a, b)
#     return z
# z = get_zip()
# z = list(z)
# print(len(z))
a = [784, 30, 10]
[np.random.rand(y, x) for x, y in zip(a[:-1], a[1:])]