from TensorForceAdapter import default_environment as env

#from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner


def main():
    max_episodes = 1000
    #max_timesteps = 1000

    network_spec = [
        #dict(type='flatten'),
        dict(type='dense', size=100, activation='tanh'),
        dict(type='dense', size=20, activation='tanh'),
        #dict(type='dense', size=32, activation='tanh'),
    ]

    agent = DQNAgent(
        states=env.states,
        actions=env.actions,
        network=network_spec,
        #batch_size=64
    )

    runner = Runner(agent, env)

    report_episodes = 10

    #global prev
    global prev
    prev = 0

    def episode_finished(r):
        global prev
        if r.episode % report_episodes == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep-prev))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
        prev = r.timestep
        #print("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run(num_episodes=max_episodes, max_episode_timesteps=None, episode_finished=episode_finished)
    runner.close()

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

if __name__ == '__main__':
    main()
