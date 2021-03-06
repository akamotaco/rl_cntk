# https://github.com/LuEE-C/PPO-Keras/blob/8c61f59339b7dae2585357ba6427037f1ceca84a/Main.py
# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

import numpy as np

import gym

# from keras.models import Model
# from keras.layers import Input, Dense
# from keras import backend as K
# from keras.optimizers import Adam

# import numba as nb
from tensorboardX import SummaryWriter

import random
from time import gmtime, strftime
import cntk as C


C.try_set_default_device(C.gpu(0))

ENV = 'LunarLander-v2'
CONTINUOUS = False

EPISODES = 100000

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
NOISE = 1.0 # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 2048
BATCH_SIZE = 256
NUM_ACTIONS = 4
NUM_STATE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


# @nb.jit
# def exponential_average(old, new, b1):
#     return old * b1 + (1-b1) * new


# def proximal_policy_optimization_loss(advantage, old_prediction):
#     def loss(y_true, y_pred):
#         prob = K.sum(y_true * y_pred, axis=-1)
#         old_prob = K.sum(y_true * old_prediction, axis=-1)
#         r = prob/(old_prob + 1e-10)
#         return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
#     return loss


# def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
#     def loss(y_true, y_pred):
#         var = K.square(NOISE)
#         pi = 3.1415926
#         denom = K.sqrt(2 * pi * var)
#         prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
#         old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

#         prob = prob_num/denom
#         old_prob = old_prob_num/denom
#         r = prob/(old_prob + 1e-10)

#         return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
#     return loss


class Agent:
    def __init__(self):
        self.critic = self.build_critic()
        if CONTINUOUS is False:
            self.actor = self.build_actor()
        else:
            self.actor = self.build_actor_continuous()

        self.env = gym.make(ENV)
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.name = self.get_name(strftime("%Y%m%d%H%M%S", gmtime()))
        self.writer = SummaryWriter(self.name)
        self.gradient_steps = 0

    def get_name(self, ext=''):
        # name = 'AllRuns/'
        name = 'log/'
        if CONTINUOUS is True:
            name += 'continous/'
        else:
            name += 'discrete/'
        name += ENV
        return name + ext

    def build_actor(self):
        # state_input = Input(shape=(NUM_STATE,))
        # advantage = Input(shape=(1,))
        # old_prediction = Input(shape=(NUM_ACTIONS,))

        # x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        # for _ in range(NUM_LAYERS - 1):
        #     x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        # out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

        # model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        # model.compile(optimizer=Adam(lr=LR),
        #               loss=[proximal_policy_optimization_loss(
        #                   advantage=advantage,
        #                   old_prediction=old_prediction)])
        # model.summary()

        # return model

        state_input = C.input_variable(NUM_STATE, name='state_input')
        advantage = C.input_variable(1, name='advantage')
        old_prediction = C.input_variable(NUM_ACTIONS, name='old_prediction')

        x = C.layers.Dense(HIDDEN_SIZE, C.tanh)(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = C.layers.Dense(HIDDEN_SIZE, C.tanh)(x)
        
        out_actions = C.layers.Dense(NUM_ACTIONS, C.softmax, name='action')(x)

        return out_actions


    def build_actor_continuous(self):
        # state_input = Input(shape=(NUM_STATE,))
        # advantage = Input(shape=(1,))
        # old_prediction = Input(shape=(NUM_ACTIONS,))

        # x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        # for _ in range(NUM_LAYERS - 1):
        #     x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        # out_actions = Dense(NUM_ACTIONS, name='output', activation='tanh')(x)

        # model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        # model.compile(optimizer=Adam(lr=LR),
        #               loss=[proximal_policy_optimization_loss_continuous(
        #                   advantage=advantage,
        #                   old_prediction=old_prediction)])
        # model.summary()

        # return model

        state_input = C.input_variable(NUM_STATE, name='state_input')
        advantage = C.input_variable(1, name='advantage')
        old_prediction = C.input_variable(NUM_ACTIONS, name='old_prediction')

        x = C.layers.Dense(HIDDEN_SIZE, C.tanh)(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = C.layers.Dense(HIDDEN_SIZE, C.tanh)(x)
        
        out_actions = C.layers.Dense(NUM_ACTIONS, C.tanh, name='actions')

        return out_actions

    def build_critic(self):

        # state_input = Input(shape=(NUM_STATE,))
        # x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        # for _ in range(NUM_LAYERS - 1):
        #     x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        # out_value = Dense(1)(x)

        # model = Model(inputs=[state_input], outputs=[out_value])
        # model.compile(optimizer=Adam(lr=LR), loss='mse')

        # return model

        state_input = C.input_variable(NUM_STATE, name='state_input')
        x = C.layers.Dense(HIDDEN_SIZE, C.tanh)(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = C.layers.Dense(HIDDEN_SIZE, C.tanh)(x)
        
        out_value = C.layers.Dense(1, name='value')(x)

        return out_value

    def reset_env(self):
        self.episode += 1
        if self.episode % 100 == 0:
            self.val = True
        else:
            self.val = False
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self):
        # p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        # if self.val is False:
        #     action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        # else:
        #     action = np.argmax(p[0])
        # action_matrix = np.zeros(NUM_ACTIONS)
        # action_matrix[action] = 1
        # return action, action_matrix, p

        # from IPython import embed;embed(header='get_action')
        p = self.actor.eval({self.actor.arguments[0]:self.observation})
        if self.val is False:
            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        return action, action_matrix, p

    def get_action_continuous(self):
        # p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        # if self.val is False:
        #     action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        # else:
        #     action = action_matrix = p[0]
        # return action, action_matrix, p

        from IPython import embed;embed(header='get_action_cont')

    def transform_reward(self):
        if self.val is True:
            self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
            # print('Val episode reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
            # print('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            if CONTINUOUS is False:
                action, action_matrix, predicted_action = self.get_action()
            else:
                action, action_matrix, predicted_action = self.get_action_continuous()
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            old_prediction = pred

#
            # from IPython import embed;embed(header='run')
            # exit()
#
            # pred_values = self.critic.predict(obs)

            # advantage = reward - pred_values

            # actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            # critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            # self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            # self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)

#region actor training
            pred_values = self.critic.eval({self.critic.arguments[0]:obs})

            advantage = reward - pred_values

            # actor_loss = 
            c_action = C.input_variable(action.shape[-1], name='action')
            c_prediction = C.input_variable(old_prediction.shape[-1], name='old_prediction')
            c_advantage =  C.input_variable(1, name='advantage')

            prob = C.reduce_sum(c_action * self.actor)
            old_prob = C.reduce_sum(c_action * c_prediction)
            ratio = prob/(old_prob + 1e-10)
            surr1 = c_advantage * ratio
            surr2 = c_advantage * C.clip(ratio, 1-LOSS_CLIPPING,  1+LOSS_CLIPPING)
            # loss = -C.reduce_mean(C.element_min(surr1, surr2) + ENTROPY_LOSS * -(prob * C.log(prob + 1e-10))) # from keras
            neglog_loss = -C.element_min(surr1, surr2)
            entropy_loss = - ENTROPY_LOSS * -(prob * C.log(prob + 1e-10))
            # loss = -C.element_min(surr1, surr2) - ENTROPY_LOSS * -(prob * C.log(prob + 1e-10)) # from keras
            # loss = -C.element_min(surr1, surr2) + ENTROPY_LOSS * -(prob * C.log(prob + 1e-10)) # from pytorch ???
            loss = C.reduce_mean(neglog_loss + entropy_loss)
            actor_loss = loss

            trainer = C.Trainer(actor_loss, (actor_loss,None),  C.adam(actor_loss.parameters, C.learning_parameter_schedule_per_sample(LR), C.learning_parameter_schedule_per_sample(0.99)))

            avg = 0
            avg_out = {neglog_loss.output: 0, entropy_loss.output:0}
            for epoch in range(EPOCHS):
                data_size = action.shape[0]
                suffle_idx = random.sample(list(range(data_size)),data_size)

                mb_action = action[suffle_idx]
                mb_obs = obs[suffle_idx]
                mb_old_prediction = old_prediction[suffle_idx]
                mb_advantage = advantage[suffle_idx]

                updated, out = trainer.train_minibatch(dict(zip(actor_loss.arguments,[mb_advantage, mb_action, mb_obs, mb_old_prediction])),
                                        outputs=[neglog_loss.output, entropy_loss.output])
                # print(trainer.previous_minibatch_loss_average)
                avg += trainer.previous_minibatch_loss_average
                avg_out[neglog_loss.output] += out[neglog_loss.output].mean()
                avg_out[entropy_loss.output] +=  out[entropy_loss.output].mean()
#endregion
            self.writer.add_scalar('Actor loss', avg/EPOCHS , self.gradient_steps)
            self.writer.add_scalar('neglog loss', avg_out[neglog_loss.output]/EPOCHS , self.gradient_steps)
            self.writer.add_scalar('entropy loss', avg_out[entropy_loss.output]/EPOCHS , self.gradient_steps)
            

#region critic training
            c_reward = C.input_variable(1, name='reward')
            loss = C.reduce_mean(C.square(self.critic - c_reward))
            critic_loss = loss

            trainer = C.Trainer(critic_loss, (critic_loss,None),  C.adam(critic_loss.parameters, C.learning_parameter_schedule_per_sample(LR), C.learning_parameter_schedule_per_sample(0.99)))

            avg = 0
            for epoch in range(EPOCHS):
                data_size = action.shape[0]
                suffle_idx = random.sample(list(range(data_size)),data_size)

                mb_obs = obs[suffle_idx]
                mb_reward = reward[suffle_idx]

                trainer.train_minibatch(dict(zip(critic_loss.arguments,[mb_obs, mb_reward])))
                # print(trainer.previous_minibatch_loss_average)
                avg += trainer.previous_minibatch_loss_average
#endregion
            self.writer.add_scalar('Critic loss', avg/EPOCHS, self.gradient_steps)

            self.gradient_steps += 1


if __name__ == '__main__':
    ag = Agent()
    ag.run()