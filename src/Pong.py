'''
Our own implementation of Pong-like game.

Everything is a global variable, so we can just import this file in Main.py and all the game
data will be accessible in the Pong.* namespace.

I'm using numpy arrays because I think they will be compatible with Tensorflow.
I'm using pygame to do the animation, which is certifiably terrible right now, but its better than nothing.

TODO:
 - Write the collisions. There is a function called "collissions" that needs to be filled out.
 - The reward function. There is a function called "reward" that needs to be filled out.
 - The expert action function is pretty bad.
 - Machine Learning

-Max
'''

import math
import numpy as np
import random
import pygame
from pygame.locals import *
import sys
from collections import namedtuple

pygame.init()
fps = pygame.time.Clock()


WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)

width = 400.0
x_center = width/2
height = 400.0
y_center = height/2

speed = 1.0

canvas = pygame.display.set_mode((round(width), round(height)), 0, 32)
pygame.display.set_caption('Hello World')

class Element:
    def __init__(self, position, velocity, ball=False):
        self.vel = velocity
        self.pos = position
        self.ball = ball

    def update(self):
        self.pos += self.vel

    def flat(self):
        return np.concatenate([self.pos, self.vel])

    def draw(self, canvas):
        pygame.draw.circle(canvas, RED, _round(self.pos), 5 if self.ball else 20)


ball = Element(np.array([x_center,y_center]), np.array([random.choice((1, -1)),random.choice((.7, -.7))]), ball=True)
player1 = Element(np.array([10.0,y_center]), np.array([0,0]))
player2 = Element(np.array([width - 10.0,y_center]), np.array([0,0]))

game = {ball, player1, player2}



StateReward = namedtuple("StateReward", ['state', 'reward'])
def step(action): # action is a velocity vector

    player1.vel = regulate_speed(action)
    player2.vel = regulate_speed(expert_action())

    collisions()

    for i in game:
        i.update()

    return StateReward(flat(), reward()) # state reward tuple

def collisions(): # needs to modify game elements' vel vectors if collision is detected
    pass # TODO: all of it

def expert_action(): # Expert is player 2 !!
    # TODO: get good
    vector = ball.pos - player2.pos
    vector[0] = 0
    return vector

def regulate_speed(action):
    if magnitude(action) > speed:
        action = speed * action / magnitude(action)
    return action

def reward(): # basically the score function
    return 0 # TODO

def flat(): # represents the entire statespace as a 1D array for learning purposes
    ret = np.array([])
    for i in game:
        ret = np.concatenate([ret, i.flat()])
    return ret

def magnitude(array):
    sum = 0
    for i in array:
        sum += i*i
    return math.sqrt(sum)

def draw():
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    canvas.fill(BLACK)
    for i in game:
        i.draw(canvas)
    pygame.display.update()
    fps.tick(60)

def _round(array):
    new = []
    for i in array:
        new.append(int(round(i)))
    return np.array(new)



def main():
    while True:
        step(np.array([1,1]))
        draw()

if __name__ == '__main__':
    main()