# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

import numpy as np
import gym
import cntk as C
from cntk_distribution import Categorical

from tensorboardX import SummaryWriter

_writer = SummaryWriter('log/discrete/LunarLander-v2_pytorch_method2_low_lr1')

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(): # (nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        state_input = C.input_variable(state_dim, name='state_input')

        # actor
        self.action_layer = C.layers.Sequential([
            C.layers.Dense(n_latent_var, C.tanh),
            C.layers.Dense(n_latent_var, C.tanh),
            C.layers.Dense(action_dim, C.softmax, name='action_prob')
            ])(state_input)
        
        # critic
        self.value_layer = C.layers.Sequential([
            C.layers.Dense(n_latent_var, C.tanh),
            C.layers.Dense(n_latent_var, C.tanh),
            C.layers.Dense(1, name='value')
            ])(state_input)
        
    def act(self, state, memory):
        action_probs = self.action_layer.eval({self.action_layer.arguments[0]:state})
        dist = Categorical(action_probs)
        action = dist.sample().eval()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action).eval())

        return int(action)
    
    def evaluate(self, action_shape): # , state, action):
        action = C.input_variable(action_shape, name='action') # old_action

        action_probs = self.action_layer #(from old_state)
        # print('input old_state')
        dist = Categorical(action_probs)

        action_logprobs =  dist.log_prob(action)
        dist_entropy =  dist.entropy()

        state_value = self.value_layer

        return action_logprobs, state_value, dist_entropy
    
    def copy_from(self, policy):
        p1, p2 = self.action_layer.parameters, policy.action_layer.parameters
        for i in range(len(p1)):
            p1[i].value = p2[i].value

        p1, p2 = self.value_layer.parameters, policy.value_layer.parameters
        for i in range(len(p1)):
            p1[i].value = p2[i].value

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var)

        # self.policy.action_layer.restore('action_layer.model')
        # self.policy.value_layer.restore('value_layer.model')

        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var)
        self.policy_old.copy_from(self.policy)
        
        self.trainer = None
        self.chunk = None

        self.loss = None

        self.gradient_steps = 0
    
    def Trainer(self):
        return self.trainer, self.chunk
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = np.array(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = memory.states
        old_actions = memory.actions
        old_logprobs = memory.logprobs
        
        avg_out = {}

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            if self.loss is None:
                # Evaluating old actions and values :
                self.logprobs, self.state_values, self.dist_entropy = self.policy.evaluate(1) #old_states, old_actions)

                self.c_old_logprobs = C.input_variable(self.logprobs.shape, name='old_log_probs')
                
                # Finding the ratio (pi_theta / pi_theta__old): # (importance sampling)
                ratios = C.exp(self.logprobs - C.stop_gradient(self.c_old_logprobs))

                self.c_rewards = C.input_variable(1, name='rewards')

                advantages = self.c_rewards - C.stop_gradient(self.state_values)
                # Finding Surrogate Loss:
                surr1 = ratios * advantages
                surr2 = C.clip(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                neglog_loss = -C.element_min(surr1, surr2)
                entropy_loss = -0.01*self.dist_entropy
                actor_loss = C.reduce_mean(neglog_loss + entropy_loss)
                critic_loss = 0.5*C.reduce_mean(C.square(self.state_values - self.c_rewards))
                self.loss = actor_loss + critic_loss

                self.chunk = {'neglog_loss':neglog_loss,
                              'entropy_loss':entropy_loss,
                              'actor_loss':actor_loss,
                              'critic_loss':critic_loss}

                self.trainer = C.Trainer(self.loss, (self.loss, None), C.adam(self.loss.parameters, C.learning_parameter_schedule_per_sample(self.lr), C.learning_parameter_schedule_per_sample(0.9)))

            actor_loss = self.chunk['actor_loss']
            critic_loss = self.chunk['critic_loss']
            neglog_loss = self.chunk['neglog_loss']
            entropy_loss = self.chunk['entropy_loss']

            updated, outs = self.trainer.train_minibatch(dict(zip(self.loss.arguments,
                                        [np.vstack(old_states).astype(np.float32),
                                        np.vstack(old_actions).astype(np.float32),
                                        np.vstack(old_logprobs).astype(np.float32),
                                        np.vstack(rewards).astype(np.float32)]
                                        )),
                                    outputs=[actor_loss.output, critic_loss.output, neglog_loss.output, entropy_loss.output])

            if neglog_loss.output not in avg_out.keys():
                avg_out[neglog_loss.output] = 0
            if entropy_loss.output not in avg_out.keys():
                avg_out[entropy_loss.output] = 0
            if actor_loss.output not in avg_out.keys():
                avg_out[actor_loss.output] = 0
            if critic_loss.output not in avg_out.keys():
                avg_out[critic_loss.output] = 0
            
            avg_out[neglog_loss.output] += outs[neglog_loss.output].mean()
            avg_out[entropy_loss.output] += outs[entropy_loss.output].mean()
            avg_out[actor_loss.output] += outs[actor_loss.output].mean()
            avg_out[critic_loss.output] += outs[critic_loss.output].mean()

        # Copy new weights into old policy:
        self.policy_old.copy_from(self.policy)

        _writer.add_scalar('Actor loss', avg_out[actor_loss.output]/self.K_epochs , self.gradient_steps)
        _writer.add_scalar('neglog loss', avg_out[neglog_loss.output]/self.K_epochs , self.gradient_steps)
        _writer.add_scalar('entropy loss', avg_out[entropy_loss.output]/self.K_epochs , self.gradient_steps)
        _writer.add_scalar('Critic loss', avg_out[critic_loss.output]/self.K_epochs, self.gradient_steps)
        self.gradient_steps += 1
        
def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = True # False # 
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002 # 1e-3
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################

    C.try_set_default_device(C.gpu(0))
    
    if random_seed:
        # torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
                
        avg_length += t

        _writer.add_scalar('Episode reward', running_reward, i_episode)
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            # torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            # ppo.policy.action_layer.save('action_layer.model')
            # ppo.policy.value_layer.save('value_layer.model')
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    
