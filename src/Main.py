import tensorflow as tf
import numpy as np
import Pong

print("Hello TensorFlow")


while True:
    state, reward = Pong.step(np.array([1, 1]))
    Pong.draw()

