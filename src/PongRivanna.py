'''
Our own implementation of Pong-like game.

Everything is a global variable, so we can just import this file in Main.py and all the game
data will be accessible in the Pong.* namespace.

I'm using numpy arrays because I think they will be compatible with Tensorflow.
I'm using pygame to do the animation, which is pretty bad right now, but its better than nothing.

TODO:
 - Figure out if we want to cap ball speed
 - Figure out the right player speed cap
 - Tune other parameters
 - Machine Learn

'''

import math
import numpy as np
import random
#import pygame
#from pygame.locals import *
#import sys
from collections import namedtuple

#from gym.spaces import Box
from GymCopies import Box

#pygame.init()
#fps = pygame.time.Clock()


WARP_SPEED = 8
COLLISION_LOOKAHEAD = 3


WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)

r2o2 = math.sqrt(2)/2
DIRECTIONS = namedtuple('Directions', ['ZERO', 'NORTH', 'NORTHEAST', 'EAST', 'SOUTHEAST', 'SOUTH', 'SOUTHWEST', 'WEST', 'NORTHWEST'])\
    (np.array([0,0]), np.array([0,1]), np.array([r2o2,r2o2]), np.array([1,0]), np.array([r2o2,-r2o2]), np.array([0,-1]),\
     np.array([-r2o2,-r2o2]), np.array([-1,0]), np.array([-r2o2,r2o2]))

width = 400.0
x_center = width/2
height = 400.0
y_center = height/2
window_vector = np.array([width, height])

speed = 2.0 # TODO: tweak
ball_speed = 10.0
score_limit = 5

# Ball, Player1, Player2
observation_space = Box(np.array([0, 0, -ball_speed, -ball_speed, 0, 0, -speed, -speed, 0,0, -speed,-speed]),
                        np.array([width, width, ball_speed, ball_speed, width, width, speed, speed, width,width, speed,speed]))
action_space = Box(np.array([-speed, -speed]), np.array([speed, speed]))

#canvas = pygame.display.set_mode((round(width), round(height)), 0, 32)
#font = pygame.font.SysFont('Comic Sans MS', 30)
#pygame.display.set_caption('Hello World')

class Element:
    def __init__(self, position, velocity, radius, mass=1.0, ball=False, color=RED):
        self.vel = velocity
        self.pos = position
        self.radius = radius
        self.mass = mass
        self.ball = ball
        self.color = color

    def update(self):
        self.pos += self.vel*WARP_SPEED

    def flat(self):
        return np.concatenate([self.pos, self.vel])

    def draw(self, canvas):
        #pygame.draw.circle(canvas, self.color, _round(self.pos), self.radius)
        pass


global ball, player1, player2, _elements, _reward
def init():
    global ball, player1, player2, _elements, _reward
    ball = Element(np.array([x_center, y_center]), np.array([random.choice((1, -1)),random.choice((.7, -.7))]), 5, mass=.5, ball=True)
    player1 = Element(np.array([20.0, y_center]), np.array([0,0]), 15, color=GREEN)
    player2 = Element(np.array([width - 20.0, y_center]), np.array([0,0]), 15)

    _elements = (ball, player1, player2)

    player1.score = 0
    player2.score = 0

    _reward = 0

init()

StateRewardDone = namedtuple("StateReward", ['state', 'reward', 'done'])

def step(action): # action is a velocity vector
    player1.vel = regulate_speed(action, speed)
    player2.vel = regulate_speed(expert_action(player2), speed)
    return _step_execute()

def simple_step(action): #action is the index of a direction unit vector in DIRECTIONS
    player1.vel = speed * DIRECTIONS[action]
    player2.vel = regulate_speed(expert_action(player2), speed)
    return _step_execute()

def _step_execute():
    collisions()
    regulate_speed(ball.vel, ball_speed)

    for i in _elements:
        i.update()

    r = _reward
    s = flat()
    if r != 0:
        upon_score() # resets _reward

    return StateRewardDone(s, r, player1.score == score_limit or player2.score == score_limit)


def reset():
    init()
    return flat()

def upon_score(): # resets _reward
    global _reward
    if _reward > 0:
        player1.score += 1
    elif _reward < 0:
        player2.score += 1
    _reward = 0
    ball.pos = np.array([x_center, y_center])
    ball.vel = np.array([random.choice((1, -1)),random.choice((.7, -.7))])
    return


def collisions():
    _check_goal()
    _physics()
    _boundaries()

def _boundaries():
    for i in _elements:
        for j in (0,1):
            if i.pos[j] + i.radius >= window_vector[j] and i.vel[j] > 0 or i.pos[j] - i.radius <= 0 and i.vel[j] < 0:
                i.vel[j] *= -1
    if player1.pos[0] > x_center and player1.vel[0] > 0:
        player1.vel[0] = 0
    if player2.pos[0] < x_center and player2.vel[0] < 0:
        player2.vel[0] = 0

def _check_goal():
    global _reward
    if ball.pos[0] - ball.radius <= 0:
        #print("score!")
        _reward = -1
    elif ball.pos[0] + ball.radius >= width:
        _reward = 1

def _physics():
    def colliding(thing1, thing2):
        vector = thing2.pos - thing1.pos
        #vector_transpose = np.transpose(vector)
        mag = magnitude(vector)
        radii = thing1.radius + thing2.radius
        vel1 = np.dot(thing1.vel, vector)
        vel2 = -1*np.dot(thing2.vel, vector)
        return mag < radii + COLLISION_LOOKAHEAD and vel1 + vel2 > 0
    def resolve(a, b):
        a.vel = a.vel - 2*b.mass/(a.mass + b.mass)*np.dot(a.vel - b.vel, a.pos - b.pos)/(magnitude(a.pos - b.pos)**2) * (a.pos - b.pos)
        b.vel = b.vel - 2*a.mass/(a.mass + b.mass)*np.dot(b.vel - a.vel, b.pos - a.pos)/(magnitude(b.pos - a.pos)**2) * (b.pos - a.pos)

    for i in range(len(_elements)):
        for j in range(i+1, len(_elements)):
            if colliding(_elements[i], _elements[j]):
                resolve(_elements[i], _elements[j])


def expert_action(element):
    if magnitude(element.pos - ball.pos) > 30:
        vector = ball.pos + 12*ball.vel - element.pos
        vector[0] = width - 20 - element.pos[0]
    else:
        vector = ball.pos - element.pos
    return vector

def regulate_speed(action, max):
    if magnitude(action) > max:
        action = max * action / magnitude(action)
    return action


def flat(): # represents the entire statespace as a 1D array for learning purposes
    ret = np.array([])
    for i in _elements:
        ret = np.concatenate([ret, i.flat()])
    return ret

def magnitude(array):
    sum = 0
    for i in array:
        sum += i*i
    return math.sqrt(sum)


def draw():
    pass
    #for event in pygame.event.get():
    #    if event.type == QUIT:
    #        pygame.quit()
    #        sys.exit()
    #canvas.fill(BLACK)
    #for i in _elements:
    #    i.draw(canvas)
    #score1 = font.render(str(player1.score), False, RED)
    #canvas.blit(score1, (20, 20))
    #score2 = font.render(str(player2.score), False, RED)
    #canvas.blit(score2, (width - 20, 20))
    #pygame.display.update()
    #fps.tick(60)


def _round(array):
    new = []
    for i in array:
        new.append(int(round(i)))
    return np.array(new)



def main():
    while True:
        step(ball.pos - player1.pos)
        #draw()

if __name__ == '__main__':
    main()
