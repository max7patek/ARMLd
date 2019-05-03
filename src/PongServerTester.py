'''
Our own implementation of Pong-like game.

'''
#.023
# import keras.backend as K
# import networkAC as ac
import math
import numpy as np
import random
import pygame
from pygame.locals import *
import sys
from collections import namedtuple
from gym.spaces import Box
import os
from multiprocessing import Pool
#import tensorflow as tf
import time, random, threading
import requests

animate = True
if animate:
    pygame.init()
    fps = pygame.time.Clock()
    font = pygame.font.SysFont('Comic Sans MS', 30)
    pygame.display.set_caption('Hello World')

StateRewardDone = namedtuple("StateReward", ['state', 'reward', 'done', 'hits'])
r2o2 = math.sqrt(2)/2
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)

class Pong:

    def __init__(self, **kwargs):
        self.WARP_SPEED = 6
        self.COLLISION_LOOKAHEAD = 3
        self.wins = 0
        self.DIMENSIONS = 3
        if 'DIMENSIONS' in kwargs:
            self.DIMENSIONS = kwargs['DIMENSIONS']

        self.DIRECTIONS =   [
                                np.array([int(i == d) for i in range(self.DIMENSIONS)])
                                    for d in range(self.DIMENSIONS)
                            ] + [
                                np.array([-1*int(i == d) for i in range(self.DIMENSIONS)])
                                    for d in range(self.DIMENSIONS)
                            ]

        # assuming that we play in a square/cube/hypercube
        self.width = 400.0
        width = self.width
        if animate:
            global canvas
            canvas = pygame.display.set_mode([int(width) for _ in range(2)])
        # self.height = 400.0
        # height = self.height

        self.speed = 2.0 # TODO: tweak
        self.ball_speed = 10.0
        self.min_ball_x_speed = .5
        self.ball_x_accel = 0.02

        self.score_limit = 5
        self.total_points = 0

        self.__dict__.update(kwargs)

        WARP_SPEED = self.WARP_SPEED
        speed = self.speed
        ball_speed = self.ball_speed
        min_ball_x_speed = self.min_ball_x_speed
        ball_x_accel = self.ball_x_accel

        self.window_vector = np.array([self.width for _ in range(self.DIMENSIONS)])
        dims = self.DIMENSIONS
        self.observation_space = Box(
            np.array(
                [0 for _ in range(dims)]+[-ball_speed for _ in range(dims)]+
                [0 for _ in range(dims)]+[-speed for _ in range(dims)]+
                [0 for _ in range(dims)]+[-speed for _ in range(dims)]
            ),
            np.array(
                [width for _ in range(dims)]+[ball_speed for _ in range(dims)]+
                [width for _ in range(dims)]+[speed for _ in range(dims)]+
                [width for _ in range(dims)]+[speed for _ in range(dims)]
            ),
        )
        self.action_space = Box(np.array([-speed for _ in range(dims)]), np.array([speed for _ in range(dims)]))

        class Element:
            def __init__(self, position, velocity, radius, mass=1.0, ball=False, color=RED, player1=False):
                self.vel = velocity
                self.pos = position
                self.radius = radius
                self.mass = mass
                self.ball = ball
                self.color = color
                self.player1 = player1

            def update(self):
                self.pos += self.vel*WARP_SPEED
                if self.ball:
                    self.vel[0] += ball_x_accel * (-1 if self.pos[0] < width/2 else 1)


            def flat(self):
                return np.concatenate([self.pos, self.vel])

            def draw(self, canvas):
                if dims == 2:
                    pygame.draw.circle(canvas, self.color, _round(self.pos), self.radius)
                elif dims > 2:
                    pygame.draw.circle(canvas, self.color, _round(self.pos)[0:2], self.radius//_positive(int(self.pos[2]/50))+1)

        self.Element = Element
        self.init()




    def init(self):
        #global ball, player1, player2, _elements, _reward

        self.ball = self.Element(
                        position=np.array([self.width/2 for _ in range(self.DIMENSIONS)]),
                        velocity=self.random_vec(),
                        radius=5, mass=.5, ball=True)
        self.player1 = self.Element(
                        position=np.array([20.0]+[self.width/2 for _ in range(self.DIMENSIONS-1)]),
                        velocity=np.array([0 for _ in range(self.DIMENSIONS)]),
                        radius=15, color=GREEN, player1=True)
        self.player2 = self.Element(
                        position=np.array([self.width - 20.0]+[self.width/2 for _ in range(self.DIMENSIONS-1)]),
                        velocity=np.array([0 for _ in range(self.DIMENSIONS)]),
                        radius=15)
        self._elements = (self.ball, self.player1, self.player2)
        self.player1.score = 0
        self.player2.score = 0
        self._reward = 0
        self.hits = 0



    def step(self, action): # action is a velocity vector
        self.player1.vel = self.regulate_speed(action, self.speed)
        self.player2.vel = self.regulate_speed(self.expert_action(self.player2), self.speed)
        return self._step_execute()

    def simple_step(self, action): #action is the index of a direction unit vector in DIRECTIONS

        self.player1.vel = self.speed * self.DIRECTIONS[action]
        self.player2.vel = self.regulate_speed(self.expert_action(self.player2), self.speed)
        return self._step_execute()

    def _step_execute(self):
        self.collisions()
        self.regulate_speed(self.ball.vel, self.ball_speed)

        for i in self._elements:
            i.update()

        r = self._reward
        s = self.flat()
        hits = self.hits
        if r != 0:
            self.upon_score() # resets _reward

        return StateRewardDone(s, r, self.player1.score == self.score_limit or self.player2.score == self.score_limit, hits)


    def reset(self):
        self.init()
        return self.flat()

    def upon_score(self): # resets _reward
        if self._reward > 0:
            if self.hits > 0:
                self.player1.score += 1
        elif self._reward < 0:
            self.player2.score += 1
        self._reward = 0
        self.ball.pos = np.array([self.width/2 for _ in range(self.DIMENSIONS)])
        self.ball.vel = self.random_vec()
        self.hits = 0
        return


    def collisions(self):
        self._check_goal()
        self._physics()
        self._boundaries()

    def _boundaries(self):
        for i in self._elements:
            for j in range(self.DIMENSIONS):
                if i.pos[j] + i.radius >= self.window_vector[j] and i.vel[j] > 0 \
                        or i.pos[j] - i.radius <= 0 and i.vel[j] < 0:
                    i.vel[j] *= -1
        median = 100
        if self.player1.pos[0] > self.width/2 - median/2 and self.player1.vel[0] > 0:
            self.player1.vel[0] = 0
            self.player1.pos[0] = self.width/2 - median/2
        if self.player2.pos[0] < self.width/2 + median/2 and self.player2.vel[0] < 0:
            self.player2.vel[0] = 0
            self.player2.pos[0] = self.width/2 + median/2

    def _check_goal(self):
        if self.ball.pos[0] - self.ball.radius <= 0:
            self._reward += -10
        elif self.ball.pos[0] + self.ball.radius >= self.width:
            self._reward += 10

    def _physics(self):
        def colliding(thing1, thing2):
            vector = thing2.pos - thing1.pos
            #vector_transpose = np.transpose(vector)
            mag = self.magnitude(vector)
            radii = thing1.radius + thing2.radius
            vel1 = np.dot(thing1.vel, vector)
            vel2 = -1*np.dot(thing2.vel, vector)
            return mag < radii + self.COLLISION_LOOKAHEAD and vel1 + vel2 > 0
        def resolve(a, b):
            a.vel = a.vel - 2*b.mass/(a.mass + b.mass)*np.dot(a.vel - b.vel, a.pos - b.pos)/(self.magnitude(a.pos - b.pos)**2) * (a.pos - b.pos)
            b.vel = b.vel - 2*a.mass/(a.mass + b.mass)*np.dot(b.vel - a.vel, b.pos - a.pos)/(self.magnitude(b.pos - a.pos)**2) * (b.pos - a.pos)

        for i in range(len(self._elements)):
            for j in range(i+1, len(self._elements)):
                if colliding(self._elements[i], self._elements[j]):
                    if self._elements[i].ball and self._elements[j].player1:
                        self._reward += 0
                        self.hits += 1
                    resolve(self._elements[i], self._elements[j])


    def expert_action(self, element):
        # return np.array([-self.speed, 0])
        if self.magnitude(element.pos - self.ball.pos) > 30:
            rollout = self.ball.pos + 12*self.ball.vel
            rollout[0] += 6 * self.ball_x_accel * (-1 if self.ball.pos[0] < self.width/2 else 1)
            vector = rollout - element.pos
            if element is self.player1:
                vector[0] = 20 - element.pos[0]
            else:
                vector[0] = self.width - 20 - element.pos[0]
        else:
            rollout = self.ball.pos + 2*self.ball.vel
            rollout[0] += self.ball_x_accel * (-1 if self.ball.pos[0] < self.width/2 else 1)
            vector = rollout - element.pos
        return vector

    def regulate_speed(self, action, max):
        if self.magnitude(action) > max:
            action = max * action / self.magnitude(action)
        return action


    def flat(self): # represents the entire statespace as a 1D array for learning purposes
        ret = np.array([])
        for i in self._elements:
            ret = np.concatenate([ret, i.flat()])
        return ret

    def magnitude(self, array):
        sum = 0
        for i in array:
            sum += i*i
        return math.sqrt(sum)

    def draw(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        canvas.fill(BLACK)
        for i in self._elements:
            i.draw(canvas)
        score1 = font.render(str(self.player1.score), False, RED)
        canvas.blit(score1, (20, 20))
        score2 = font.render(str(self.player2.score), False, RED)
        canvas.blit(score2, (self.width - 20, 20))
        pygame.display.update()
        fps.tick(120)

    def random_vec(self):
        out = np.array([random.random()*2 - 1 for _ in range(self.DIMENSIONS)])
        if abs(out[0]) < self.min_ball_x_speed:
            out[0] = self.min_ball_x_speed * (-1 if out[0] < 0 else 1)
        return out

def _round(array):
    new = []
    for i in array:
        new.append(int(round(i)))
    return np.array(new)

def _positive(num):
    return num if num > 0 else 1



SEND_EVERY = 30

def main():
    assert len(sys.argv) == 2, 'usage: pass in server URL:PORT. e.g. python3 __.py 127.0.0.1:8000'
    game = Pong(DIMENSIONS=3, WARP_SPEED=1)
    i = 0
    while True:
        i += 1
        s, _, _, _ = game.step(game.expert_action(game.player1))
        game.draw()
        if i % SEND_EVERY == 0:
            
            print(requests.post(url='http://'+sys.argv[1], data={'data':','.join(map(str, list(s) + [0]))}).text)



if __name__ == '__main__':
    main()
