import tensorflow as tf
import numpy as np
import Pong
import random

print("Hello TensorFlow")


array = np.array([1,2,3])
print(array)
print(array.reshape([1,3]))


### This doesn't work, it's just an example I found online that I tried to adapt.
### If you want to see the Pong simulation in action, run Pong.py as main.


statespace_size = len(Pong.flat())
actionspace_size = len(Pong.DIRECTIONS)
#actionspace_size = 1

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,statespace_size],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([statespace_size, actionspace_size],0,0.01)) #weights
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,actionspace_size],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)


#init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 20000
#create lists to contain total rewards and steps per episode
jList = []
rList = []

episode_count = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = Pong.reset()
        rAll = 0
        d = False
        j = 0
        print(i)
        #The Q-Network
        while j < 200:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            #a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(statespace_size)[s]})
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: s.reshape([1,statespace_size])})
            #print(a)
            if np.random.rand(1) < e:
                a[0] = random.choice(range(len(Pong.DIRECTIONS)))
            #Get new state and reward from environment
            s1,r,d = Pong.simple_step(a[0])
            if i > num_episodes - 200:
                Pong.draw()
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:s.reshape([1,statespace_size])})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1: s.reshape([1,statespace_size]),nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        #jList.append(j)
        #rList.append(rAll)


