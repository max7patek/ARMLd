import keras
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class network:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.load_model = False
        self.discount_factor = 0.99

        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.005

        self.actor = self.actor_model()
        self.critic = self.critic_model()

        if self.load_model:
            self.actor.load_weights("/Users/Sami/Documents/Folder/Coding/Python/Machine-Learning/Pong/Weights/network_actor.h5")
            self.critic.load_weights("/Users/Sami/Documents/Folder/Coding/Python/Machine-Learning/Pong/Weights/network_critic.h5")

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def actor_model(self):
        model = Sequential()
        model.add(Dense(24, activation = 'relu', input_dim = self.state_size,kernel_initializer='he_uniform'))
        model.add(Dense(48, activation = 'relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(self.value_size, activation = 'softmax',kernel_initializer='he_uniform'))
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=self.learning_rate_actor))
        return model

    def critic_model(self):
        model = Sequential()
        model.add(Dense(24, activation = 'relu', input_dim = self.state_size,kernel_initializer='he_uniform'))
        model.add(Dense(48, activation = 'relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(self.value_size, activation = 'linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate_critic))
        return model

    def _act(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def _train(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)


    def _save(self):
        self.actor.save_weights("/Users/Sami/Documents/Folder/Coding/Python/Machine-Learning/Pong/Weights/network_actor1.h5")
        self.critic.save_weights("/Users/Sami/Documents/Folder/Coding/Python/Machine-Learning/Pong/Weights/network_critic1.h5")
