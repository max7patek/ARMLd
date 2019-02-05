import keras
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import model_from_json
import random
import os
class DQNagent:

    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.eDecay = 0.995
        self.eMin = 0.01
        self.learningRate = 0.001

    def _buildModel(self,state_size,action_size):
        self.stateSize = state_size
        self.actionSize = action_size
        model = Sequential()
        model.add(Dense(60, input_dim=self.stateSize, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(self.actionSize, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learningRate))
        self.model = model

    def recall(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.actionSize)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


    def replay(self, batchSize):
        miniBatch = random.sample(self.memory, batchSize)
        for state, action, reward, next_state, done in miniBatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon>self.eMin:
                self.epsilon *= self.eDecay
