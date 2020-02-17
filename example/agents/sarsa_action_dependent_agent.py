import numpy as np

from clgridworld.grid_world_builder import GridWorldBuilder, InitialStateParams
from example.agent_trainer import AgentTrainer
from example.agents.agent import Agent
from example.agents.policy import Policy, EpsGreedy, EpsAnnealed
from clgridworld.state.state import GridWorldObservationSpace, GridWorldState
from clgridworld.state.state_factory import GridWorldStateFactory
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

class SarsaAgent(Agent):

    def __init__(self, action_space, policy: Policy, discount_factor=0.95, learning_rate=0.01, seed=0):

        self.num_actions = action_space.n
        self.policy = policy
        self.test_policy = EpsGreedy(0)

        np.random.seed(seed)

        self.Q = {}
        self.action_space = action_space

        self.gamma = discount_factor
        self.alpha = learning_rate

        self.prev_sample = None

    def get_action(self, curr_state):

        self.init_state(curr_state)

        return self.policy.select_action(self.Q[curr_state])

    def get_best_action(self, curr_state):

        self.init_state(curr_state)

        return self.test_policy.select_action(self.Q[curr_state])

    def update(self, prev_state, action, curr_state, reward):
        if self.prev_sample is not None:
            prev_sample = self.prev_sample
            self._update(prev_sample[0], prev_sample[1], prev_sample[2], prev_sample[3], action)

        self.prev_sample = (prev_state, action, curr_state, reward)

    def _update(self, prev_state, prev_action, curr_state, prev_reward, curr_action):

        self.init_state(curr_state)

        self.Q[prev_state][prev_action] += self.alpha * (prev_reward + self.gamma * self.Q[curr_state][curr_action]
                                                         - self.Q[prev_state][prev_action])

    def init_state(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros([self.num_actions])

    def inc_episode(self):
        self.policy.inc_episode()
        self.test_policy.inc_episode()

        if self.prev_sample is not None:
            prev_sample = self.prev_sample
            self._update(prev_sample[0], prev_sample[1], prev_sample[2], prev_sample[3],
                         self.get_best_action(prev_sample[2]))

def transfer_key_q_values(source_init_param, source_q_vales, target_param, target_agent, reward_ind):

    #below, upper, left, right
    source_near_locs = []
    source_near_locs.append((source_init_param[reward_ind][0] - 1, source_init_param[reward_ind][1]))
    source_near_locs.append((source_init_param[reward_ind][0] + 1, source_init_param[reward_ind][1]))
    source_near_locs.append((source_init_param[reward_ind][0], source_init_param[reward_ind][1] - 1))
    source_near_locs.append((source_init_param[reward_ind][0], source_init_param[reward_ind][1] + 1))

    target_near_locs = []
    target_near_locs.append((target_param[reward_ind][0] - 1, target_param[reward_ind][1]))
    target_near_locs.append((target_param[reward_ind][0] + 1, target_param[reward_ind][1]))
    target_near_locs.append((target_param[reward_ind][0], target_param[reward_ind][1] - 1))
    target_near_locs.append((target_param[reward_ind][0], target_param[reward_ind][1] + 1))

    for state in pre_agent_state_vals:
        for index, s_loc in enumerate(source_near_locs):
            if source_init_param[reward_ind] == state[reward_ind] and s_loc == state[1]:
                # Generate corresponding state
                n_state = GridWorldStateFactory.create(target_param.shape, target_near_locs[index], target_param.key,
                                                       target_param.lock, target_param.pit_start, target_param.pit_end)
                target_agent.Q[n_state] = source_q_vales[state]



if __name__ == '__main__':

    seed = 0

    # Generate curriculums
    ad_params = []
    ad_params.append(InitialStateParams(shape=(5, 5), player=(4, 4), key=(0, 0)))
    ad_params.append(InitialStateParams(shape=(10, 10), player=(9, 9), key=(0, 0)))
    ad_params.append(InitialStateParams(shape=(10, 10), player=(1, 4), key=(7, 5), lock=(1, 1), pit_start=(4, 2),
                                  pit_end=(4, 7)))

    pre_agent = None
    source_param = None

    for index, param in enumerate(ad_params):
        env = GridWorldBuilder.create(param)
        policy = EpsGreedy(0.1)
        agent = SarsaAgent(env.action_space, policy, discount_factor=1, seed=seed)

        if pre_agent is not None:
            pre_agent_state_vals = pre_agent.Q
            if source_param[2] is not None and param[2] is not None:
                transfer_key_q_values(source_param, pre_agent_state_vals, param, agent, 2)

            if source_param[3] is not None and param[3] is not None:
                transfer_key_q_values(source_param, pre_agent_state_vals, param, agent, 3)

        eps_rewards = AgentTrainer(env, agent).train(seed, num_episodes=2500, max_steps_per_episode=10000,
                                          episode_log_interval=100)

        pre_agent = agent
        source_param = param

