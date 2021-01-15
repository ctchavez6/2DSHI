import numpy as np

deser_test = np.load('test.npy')

print(deser_test)
print("Matrix 1: Shape = ", deser_test.shape, " type = ", type(deser_test[0][0]))
