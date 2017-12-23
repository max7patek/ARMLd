import tensorflow as tf
import numpy as np
import Pong

print("Hello TensorFlow")


while True:
    state, reward = Pong.step(Pong.ball.pos - Pong.player1.pos)
    Pong.draw()

