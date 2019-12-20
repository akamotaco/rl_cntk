# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# import torch
# import torch.nn as nn
# from torch.distributions import Categorical
import numpy as np
import gym
import cntk as C
from cntk_distribution import Categorical

from tensorboardX import SummaryWriter

_writer = SummaryWriter('log/discrete/LunarLander-v2_pytorch_method2_low_lr')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # super(ActorCritic, self).__init__()
        # self.affine = nn.Linear(state_dim, n_latent_var)
        # self.affine = C.layers.Dense(n_latent_var)(C.placeholder(state_dim)) # not used?
        
        # actor
        # self.action_layer = nn.Sequential(
        #         nn.Linear(state_dim, n_latent_var),
        #         nn.Tanh(),
        #         nn.Linear(n_latent_var, n_latent_var),
        #         nn.Tanh(),
        #         nn.Linear(n_latent_var, action_dim),
        #         nn.Softmax(dim=-1)
        #         )
        self.action_layer = C.layers.Sequential([
            C.layers.Dense(n_latent_var, C.tanh),
            C.layers.Dense(n_latent_var, C.tanh),
            C.layers.Dense(action_dim, C.softmax, name='action_prob')
            ])(C.input_variable(state_dim, name='action_layer'))
        
        # critic
        # self.value_layer = nn.Sequential(
        #         nn.Linear(state_dim, n_latent_var),
        #         nn.Tanh(),
        #         nn.Linear(n_latent_var, n_latent_var),
        #         nn.Tanh(),
        #         nn.Linear(n_latent_var, 1)
        #         )
        self.value_layer = C.layers.Sequential([
            C.layers.Dense(n_latent_var, C.tanh),
            C.layers.Dense(n_latent_var, C.tanh),
            C.layers.Dense(1, name='value')
            ])(C.input_variable(state_dim, name='value_layer'))
        
#     def forward(self):
#         raise NotImplementedError
        
    def act(self, state, memory):
#         state = torch.from_numpy(state).float().to(device) 
#         action_probs = self.action_layer(state)
#         dist = Categorical(action_probs)
#         action = dist.sample()
        
#         memory.states.append(state)
#         memory.actions.append(action)
#         memory.logprobs.append(dist.log_prob(action))

#         return action.item()

        action_probs = self.action_layer.eval({self.action_layer.arguments[0]:state})
        dist = Categorical(action_probs)
        # dist = Categorical(self.action_layer).eval({self.action_layer.arguments[0]:state})
        action = dist.sample().eval()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action).eval())

        # return np.argmax(action)
        return int(action)
    
    def evaluate(self, action_shape): # , state, action):
#         action_probs = self.action_layer(state)
#         dist = Categorical(action_probs)
        
#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
        
#         state_value = self.value_layer(state)
        
#         return action_logprobs, torch.squeeze(state_value), dist_entropy
        # action_probs = self.action_layer.eval({self.action_layer.arguments[0]:state})
        # dist = Categorical(action_probs)

        # action_logprobs =  dist.log_prob(action)
        # dist_entropy =  dist.entropy()

        # state_value = self.value_layer.eval({self.value_layer.arguments[0]:state})
        # return action_logprobs, state_value, dist_entropy

        action = C.input_variable(action_shape, name='action') # old_action

        action_probs = self.action_layer #(from old_state)
        # print('input old_state')
        dist = Categorical(action_probs)

        action_logprobs =  dist.log_prob(action)
        dist_entropy =  dist.entropy()

        state_value = self.value_layer

        return action_logprobs, state_value, dist_entropy
    
    def copy_from(self, policy):
        # from IPython import embed;embed(header='clone')
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
        
        # self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        # self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var)
        # self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.copy_from(self.policy)
        
        # self.MseLoss = nn.MSELoss()
    
        self.trainer = None
        self.chunk = None

        self.gradient_steps = 0
    
    def Trainer(self):
        return self.trainer, self.chunk
    
    # def Loss(self, c_ratios, c_state_values, c_dist_entropy):
    #     # c_ratios = C.input_variable(1, name='ratios')
    #     # c_state_values = C.input_variable(1, name='state_values')
    #     # c_dist_entropy = C.input_variable(1, name='dist_entropy')
    #     c_rewards = C.input_variable(1, name='rewards')

    #     advantages = c_rewards - c_state_values
    #     surr1 = c_ratios * advantages
    #     surr2 = C.clip(c_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
    #     # loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
    #     # loss = -C.element_min(surr1, surr2) +  0.5*C.reduce_mean(C.square(c_state_values- c_rewards)) -0.01*c_dist_entropy
    #     neglog_loss = -C.element_min(surr1, surr2)
    #     entropy_loss = -0.01*c_dist_entropy
    #     actor_loss = C.reduce_mean(neglog_loss + entropy_loss)
    #     critic_loss = 0.5*C.reduce_mean(C.square(c_state_values- c_rewards))
    #     self.loss = actor_loss + critic_loss

    #     # trainer = C.Trainer(loss, (loss, None), C.adam(loss.parameters, C.learning_parameter_schedule_per_sample(self.lr), C.learning_parameter_schedule_per_sample(0.9)))
    #     trainer = C.Trainer(self.loss, (self.loss, None), C.adam(self.loss.parameters, C.learning_parameter_schedule_per_sample(self.lr), C.learning_parameter_schedule_per_sample(0.9)))
    #     # trainer = C.Trainer(loss, (loss, None), C.adam(loss.parameters, C.learning_parameter_schedule(1e-1), C.learning_parameter_schedule(0.9)))
    #     self.trainer = trainer
        
    #     self.chunk = {
    #         'action': self.policy.action_layer.arguments[0], # old_states
    #         'value': self.policy.value_layer.arguments[0], # old_actions
    #         # 'ratios': c_ratios,
    #         # 'state_values': c_state_values,
    #         'rewards': c_rewards,
    #         # 'dist_entropy': c_dist_entropy

    #         'neglog_loss': neglog_loss,
    #         'entropy_loss': entropy_loss,
    #         'actor_loss': actor_loss,
    #         'critic_loss': critic_loss
    #     }
#        from  IPython import embed;embed(header='loss')

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
        # rewards = torch.tensor(rewards).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = np.array(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
    #     old_states = torch.stack(memory.states).to(device).detach()
    #     old_actions = torch.stack(memory.actions).to(device).detach()
    #     old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        old_states = memory.states
        old_actions = memory.actions
        old_logprobs = memory.logprobs
        
        avg_out = {}

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(1) #old_states, old_actions)

            c_old_logprobs = C.input_variable(logprobs.shape, name='old_log_probs')
            
            # Finding the ratio (pi_theta / pi_theta__old):
    #         ratios = torch.exp(logprobs - old_logprobs.detach())
            ratios = C.exp(logprobs - C.stop_gradient(c_old_logprobs))

            # Finding Surrogate Loss:
    #         advantages = rewards - state_values.detach()
    #         surr1 = ratios * advantages
    #         surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
    #         loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            # advantages = rewards - state_values
            # surr1 = ratios * advantages
            # surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            # if self.Trainer()[0] is None:
            #     print("zxcv")
            #     ll = self.Loss(ratios, state_values, dist_entropy)

    #         self.loss.eval(dict(zip(self.loss.arguments,[np.vstack(old_states),np.vstack(old_actions),n
    # ...: p.vstack(old_logprobs),np.vstack(rewards),np.vstack(old_states)])))
#            from IPython import embed;embed(header='train')
            c_rewards = C.input_variable(1, name='rewards')

            # c_rewards_mean = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            # c_rewards_mean = rewards.mean()
            # c_rewards_std = C.sqrt(C.reduce_mean(C.square(c_rewards-c_rewards_mean)))
            # c_rewards_norm = (c_rewards-c_rewards_mean)/(c_rewards_std+1e-5)

            advantages = c_rewards - C.stop_gradient(state_values)
            surr1 = ratios * advantages
            surr2 = C.clip(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            # loss = -C.element_min(surr1, surr2) +  0.5*C.reduce_mean(C.square(c_state_values- c_rewards)) -0.01*c_dist_entropy
            neglog_loss = -C.element_min(surr1, surr2)
            entropy_loss = -0.01*dist_entropy
            actor_loss = C.reduce_mean(neglog_loss + entropy_loss)
            # actor_loss = neglog_loss + entropy_loss
            critic_loss = 0.5*C.reduce_mean(C.square(state_values - c_rewards))
            loss = actor_loss + critic_loss

            trainer = C.Trainer(loss, (loss, None), C.adam(loss.parameters, C.learning_parameter_schedule_per_sample(self.lr), C.learning_parameter_schedule_per_sample(0.9)))
            # trainer = C.Trainer(loss, (loss, None), C.adam(loss.parameters, C.learning_parameter_schedule(self.lr), C.learning_parameter_schedule(0.9)))
            # trainer = C.Trainer(loss, (loss, None), C.adam(loss.parameters, C.learning_parameter_schedule(1e-1), C.learning_parameter_schedule(0.9)))
#             trainer, chunk = self.Trainer()
#             # c_ratios = chunk['ratios']
#             # c_state_values = state_values
#             c_rewards = chunk['rewards']
#             # c_dist_entropy = dist_entropy
#             c_a = chunk['action']
#             c_v = chunk['value']

# #
#             neglog_loss = chunk['neglog_loss']
#             entropy_loss = chunk['entropy_loss']
#             actor_loss = chunk['actor_loss']
#             critic_loss = chunk['critic_loss']
#
            # trainer.train_minibatch({
            #     c_v: np.vstack(old_states),
            #     c_a: np.vstack(old_actions),
            #     c_old_logprobs: np.vstack(old_logprobs),
            #     c_rewards: rewards,
            # })
            # from IPython import embed;embed(header='train')
            # self.loss.eval(dict(zip(self.loss.arguments,[np.vstack(old_states),np.vstack(old_actions),np.vstack(old_logprobs),np.vstack(rewards),np.vstack(old_states)])))
            # from IPython import embed;embed()
            # trainer.train_minibatch(dict(zip(self.loss.arguments,[np.vstack(old_states),np.vstack(old_actions),np.vstack(old_logprobs),np.vstack(rewards),np.vstack(old_states)])))
            updated, outs = trainer.train_minibatch(dict(zip(loss.arguments,[np.vstack(old_states),np.vstack(old_actions),np.vstack(old_logprobs),np.vstack(rewards),np.vstack(old_states)])),
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

    #         # take gradient step
    #         self.optimizer.zero_grad()
    #         loss.mean().backward()
    #         self.optimizer.step()
        
    #     # Copy new weights into old policy:
    #     self.policy_old.load_state_dict(self.policy.state_dict())
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
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 1e-4 #0.002
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
    
