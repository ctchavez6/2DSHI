import numpy as np
import io
import json
serialize = True
deserialize = False

test_array_to_ser = np.array([[float(1.2), 2.432, 3], [4, 5, 6]], dtype=np.float32)
test_array_to_ser = test_array_to_ser.astype(np.float32)
print("Matrix 1: Shape = ", test_array_to_ser.shape, " type = ", type(test_array_to_ser[0][0]))

print(test_array_to_ser)


a = test_array_to_ser# any NumPy array


np.save('test.npy', test_array_to_ser)

