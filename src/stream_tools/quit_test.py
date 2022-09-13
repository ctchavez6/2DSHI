import time

counter = 0

while counter < 3:
    time.sleep(5)
    counter += 1
    print("counter: ", counter)

print("done")