# https://www.cntk.ai/pythondocs/CNTK_203_Reinforcement_Learning_Basics.html

import numpy as np
import math, random
import gym

import cntk as C


class Brain:
    def __init__(self, hidden=64):
        self.model, self.trainer, self.loss = self._create(hidden)

    def _create(self, hidden):
        observation = C.input_variable(STATE_COUNT, name="s")
        q_target = C.input_variable(ACTION_COUNT, name="q")

        model = C.layers.Dense(hidden, activation=C.relu)(observation)
        model = C.layers.Dense(ACTION_COUNT)(model)

        # loss='mse'
        loss = C.reduce_mean(C.square(model - q_target)) #, axis=0)

        # optimizer
        lr = 0.00025
        lr_schedule = C.learning_parameter_schedule(lr)
        learner = C.sgd(model.parameters, lr_schedule, gradient_clipping_threshold_per_sample=10)
        trainer = C.Trainer(model, (loss, None), learner)

        return model, trainer, loss

    def train(self, x, y, epoch=1, verbose=0):
        arguments = dict(zip(self.loss.arguments, [x,y]))
        updated, results = self.trainer.train_minibatch(arguments, outputs=[self.loss.output])

    def predict(self, s):
        return self.model.eval(s)

class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

class Agent:
    def __init__(self):
        self.brain = Brain(64)
        self.memory = Memory(MEMORY_CAPACITY)
        self.steps = 0
        self.epsilon = MAX_EPSILON

    def exploitation(self, s):
        return np.argmax(self.brain.predict(s))
    def exploration(self, s):
        return random.randint(0, ACTION_COUNT-1)

    def act(self, s):
        if random.random() < self.epsilon:
            return self.exploration(s)
        else:
            return self.exploitation(s)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(STATE_COUNT)

        states = np.array([ ob[0] for ob in batch ], dtype=np.float32)
        states_ = np.array([(no_state if ob[3] is None else ob[3]) for ob in batch ], dtype=np.float32)

        q = agent.brain.predict(states)
        q_ = agent.brain.predict(states_)

        x = np.zeros((batchLen, STATE_COUNT)).astype(np.float32)
        y = np.zeros((batchLen, ACTION_COUNT)).astype(np.float32)

        for i in range(batchLen):
            s, a, r, s_ = batch[i]

            e = q[i]
            if s_ is None: # end of game
                e[a] = r
            else:
                e[a] = r + GAMMA * np.amax(q_[i])

            x[i] = s
            y[i] = e

        self.brain.train(x, y)

def training(env, agent):
    episode_number = 0
    reward_sum = 0
    while episode_number < TOTAL_EPISODES:

        s = env.reset()
        done = False
        while not done:
            # env.render()
                
            a = agent.act(s.astype(np.float32))
            s_, r, done, info = env.step(a)

            if done:
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            reward_sum += r

        episode_number += 1
        if episode_number % BATCH_SIZE_BASELINE == 0:
            print('Episode: %d, Average reward for episode %f.' % (episode_number,
                                                                reward_sum / BATCH_SIZE_BASELINE))
            if reward_sum / BATCH_SIZE_BASELINE > REWARD_TARGET:
                print('Task solved in %d episodes' % episode_number)
                break
            reward_sum = 0

def test(env, agent, render=False):
    num_episodes = 10

    for i_episode in range(num_episodes):
        ob = env.reset()
        done = False
        reward_sum = 0
        while not done:
            if render is True:
                env.render()
            action = np.argmax(agent.brain.predict(ob.astype(np.float32)))
            ob, reward, done, info = env.step(action)
            reward_sum += reward
        print(f'epi:{i_episode}\t reward_sum:{reward_sum}')

if __name__ == '__main__':
    from gym import wrappers, logger

    # C.try_set_default_device(C.gpu(0))

    env = gym.make('CartPole-v0')
    isFast = True # False # 
    isTraining = True # False # 
    isSave = False # True #
    isRender = False # True # 
    isPreset = None # 'dqn_cntk.model' # 

    #region config
    STATE_COUNT  = env.observation_space.shape[0]
    ACTION_COUNT = env.action_space.n
    print(STATE_COUNT, ACTION_COUNT)

    # Targetted reward
    REWARD_TARGET = 30 if isFast else 200
    # Averaged over these these many episodes
    BATCH_SIZE_BASELINE = 20 if isFast else 50
    MEMORY_CAPACITY = 100000
    BATCH_SIZE = 64

    TOTAL_EPISODES = 2000 if isFast else 3000

    GAMMA = 0.99 # discount factor

    MAX_EPSILON = 1
    MIN_EPSILON = 0.01 # stay a bit curious even when getting old
    LAMBDA = 0.0001    # speed of decay
    #endregion

    agent = Agent()

    if isPreset is not None:
        agent.brain.model.restore(isPreset)
    if isTraining:
        training(env,agent)
    test(env,agent, render=isRender)
    if isSave:
        agent.brain.model.save('dqn_cntk.model')
    env.close()
    exit()

#region play
