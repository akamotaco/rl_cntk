# https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
import gym
import numpy as np

import cntk as C

class Brain:
    def __init__(self, input_dim, output_dim, hidden_dims=[16,16]):
        self.model, self.loss, self.trainer = self._create_model(input_dim,output_dim,hidden_dims)
    
    def _create_model(self, input_dim, output_dim, hidden_dims):
        c_in = C.input_variable(input_dim, name='state')
        model = c_in

        for h in hidden_dims:
            model = C.layers.Dense(h, activation=C.relu)(model)
        model = C.layers.Dense(output_dim,activation=C.softmax)(model)

        c_action_prob = model
        c_action_onehot = C.input_variable(output_dim, name='action_onehot')
        c_reward = C.input_variable(1, name='reward')
        action_prob = C.reduce_sum(c_action_prob * c_action_onehot)
        log_action_prog = C.log(action_prob)
        loss = -log_action_prog * c_reward
        loss = C.reduce_mean(loss)

        lr = 1e-2
        lr_schedule = C.learning_parameter_schedule(lr)
        learner = C.adam(model.parameters, lr_schedule, C.momentum_schedule(0.9))
        trainer = C.Trainer(model,(loss,None),learner)

        return model, loss, trainer
    
    def predict(self, state):
        return self.model.eval({self.model.arguments[0]:state})

    def train(self, memory):
        S, A, R = memory[0], memory[1], memory[2]
        self.trainer.train_minibatch(dict(zip(self.loss.arguments,[S,A,R])))
        # exit()

class Agent:
    def __init__(self, input_dim, output_dim, hidden_dims=[16,16]):
        self.brain = Brain(input_dim,output_dim,hidden_dims)
        self.output_dim = output_dim
    
    def _compute_discounted_R(self, R, discount_rate=.99):
        discounted_r = np.zeros_like(R, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(R))):
            running_add = running_add * discount_rate + R[t]
            discounted_r[t] = running_add
        discounted_r -= discounted_r.mean() / discounted_r.std()
        return discounted_r.reshape(-1,1)
    
    def act(self, state):
        # return np.argmax(self.brain.predict(state))
        action_prob = self.brain.predict(state)[0]
        return np.random.choice(np.arange(self.output_dim), p=action_prob)
    
    def train(self, memory):
        S = np.array([m[0] for m in memory], np.float32)
        A = np.array([m[1] for m in memory], np.float32)
        R = np.array([m[2] for m in memory], np.float32)

        action_onehot = C.one_hot(A,num_classes=self.output_dim).eval()
        discount_reward = self._compute_discounted_R(R)

        self.brain.train([S, action_onehot, discount_reward])
        
    


def training(env, agent:Agent):
    for episode in range(300):
        done = False
        s = env.reset()
        reward_sum = 0
        memory = []

        while not done:
            a = agent.act(s)
            s_, r, done, info = env.step(a)
            reward_sum += r

            memory.append((s,a,r))

            s = s_
            if done:
                agent.train(memory)
        print(f'ep:{episode}\t r:{reward_sum}')
    return reward_sum


def test(env, agent:Agent, render=False):
    for episode in range(10):
        done = False
        s = env.reset()
        reward_sum = 0

        while not done:
            if render:
                env.render()
            a = agent.act(s)
            s_, r, done, info = env.step(a)
            reward_sum += r

            s = s_

        print(f'ep:{episode}\t r:{reward_sum}')
    return reward_sum

def main():
    # C.try_set_default_device(C.gpu(0))

    env = gym.make('CartPole-v0')
    isTraining = True # False #
    isSave = False # True # 
    isRender = False # True # 
    isPreset = None # 'pg_cntk.model'

    #region config
    state_count  = env.observation_space.shape[0]
    action_count = env.action_space.n

    agent = Agent(state_count, action_count)

    if isPreset is not None:
        agent.brain.model.restore(isPreset)

    if isTraining:
        training(env, agent)
    test(env, agent, render=isRender)
    if isSave:
        agent.brain.model.save('pg_cntk.model')
    env.close()
    exit()


if __name__ == '__main__':
    main()
