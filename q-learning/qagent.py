#!python3

import math
import os
import time
from typing import Callable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

import gym
from gym import logger
from gym.envs.registration import register

# %%
def clear_function():
    try:
        __IPYTHON__
        return lambda: clear_output(wait=True)
    except NameError:
        if os.name == 'nt':
            return lambda: os.system('cls')
        else:
            return lambda: os.system('clear')

clear = clear_function()

# %%
def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length=100, fill: str = '█', unfill: str = '-', min_filled_length: int = 0, print_percent: bool = True, print_end: str = os.linesep):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = round(100 * iteration / total, decimals)
    filled_length = max(int(length * iteration // total), min_filled_length)
    bar = fill * filled_length + unfill * (length - filled_length)
    print(f'\r{prefix} [{bar}]', end='')
    if print_percent:
        print(f' {percent}%', end='')
    print(f' {suffix}', end=print_end)

# %%
class Agent():
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = action_space

    def _print_details_(self, *_) -> None:
        pass

    def _print_progress_bar_(self, *_) -> None:
        pass

    def step(self, *_) -> np.array:
        raise NotImplementedError

    def train(self, *_) -> None:
        pass

# %%
class RandomAgent(Agent):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__(observation_space, action_space)

    def step(self, *_) -> np.array:
        return self.action_space.sample()

# %%
class DiscreteAgent(Agent):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, discretization_size: list = [20, 20]):
        super().__init__(observation_space, action_space)
        self.observation_size, self.observation_step = self._discretize_space_(observation_space, discretization_size[0])
        self.action_size,      self.action_step      = self._discretize_space_(action_space, discretization_size[1])

    def _discretize_space_(self, space: gym.Space, discretization_size: list) -> list:
        if type(space) is gym.spaces.discrete.Discrete:
            size = [space.n]
            discretization_step = None
        else:
            size = np.array([discretization_size] * np.prod(space.shape))
            discretization_step = (space.high - space.low) / (size - 1)
        return tuple(size), discretization_step

    def _get_continuous_action_(self, action: np.array) -> Tuple:
        return self._get_continuous_space_value_(action, self.action_space, self.action_step)

    def _get_continuous_observation_(self, observation: np.array) -> Tuple:
        return self._get_continuous_space_value_(observation, self.observation_space, self.observation_step)

    def _get_continuous_space_value_(self, discrete_space_value: Tuple, space: gym.Space, discretization_step: np.array) -> Tuple:
        if type(space) is gym.spaces.discrete.Discrete:
            space_value = np.asscalar(discrete_space_value)
        else:
            space_value = np.array(space.low + discrete_space_value * discretization_step)
        return space_value

    def _get_discrete_action_(self, action: np.array) -> Tuple:
        return self._get_discrete_space_value_(action, self.action_space, self.action_step)

    def _get_discrete_observation_(self, observation: np.array) -> Tuple:
        return self._get_discrete_space_value_(observation, self.observation_space, self.observation_step)

    def _get_discrete_space_value_(self, space_value: np.array, space: gym.Space, discretization_step: np.array) -> Tuple:
        if type(space) is gym.spaces.discrete.Discrete:
            discrete_space_value = (space_value,)
        else:
            discrete_space_value = ((space_value - space.low) / discretization_step).astype(np.int)
        return tuple(discrete_space_value)

# %%
class QAgent(DiscreteAgent, RandomAgent):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, discount_factor: float = 0.95, learning_rate: float = 0.01, exploration_decay_rate: float = 0.99, min_exploration_rate = 0.1, discretization_size: list = [20, 20]):
        super().__init__(observation_space, action_space, discretization_size)
        self.discount_factor        = discount_factor
        self.learning_rate          = learning_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate   = min_exploration_rate
        self.exploration_rate       = 1
        self.q_table                = np.zeros(self.observation_size + self.action_size)

    def step(self, observation: np.array,ep) -> np.array:
        discrete_observation = self._get_discrete_observation_(observation)
        noisy_q_row          = self.q_table[discrete_observation] + self.exploration_rate * np.random.randn(*self.action_size)
        discrete_action      = np.argmax(noisy_q_row)
        action               = self._get_continuous_action_(discrete_action)
        return action

    def train(self, observation: np.array, action: np.array, next_observation: np.array, reward: float, done: bool, goal_reached: bool, batch_goals_ratio: float) -> None:
        # Discretize spaces
        discrete_observation      = self._get_discrete_observation_(observation)
        discrete_action           = self._get_discrete_action_(action)
        next_discrete_observation = self._get_discrete_observation_(next_observation)
        # Learning rule
        q_best_next = np.max(self.q_table[next_discrete_observation])
        q_objective = reward + self.discount_factor * q_best_next  # Bellman's equation approximation
        error       = q_objective - self.q_table[discrete_observation + discrete_action]
        self.q_table[discrete_observation + discrete_action] += self.learning_rate * error
        if done:
            min_exploration_rate  = (1 - batch_goals_ratio) * self.min_exploration_rate
            self.exploration_rate = max(0, self.exploration_rate * self.exploration_decay_rate, min_exploration_rate)
            self.exploration_rate = min(1, self.exploration_rate)

    def _print_details_(self) -> None:
        print(f'Exploration rate = {self.exploration_rate}')
        if np.prod(self.q_table.shape) < 200:
            print('Q-table:')
            print(self.q_table)

    def _print_progress_bar_(self) -> None:
        print_progress_bar(1 - self.exploration_rate, 1, prefix='Exploitation ',
                           suffix=f'Exploration (rate = {round(self.exploration_rate, 3)})', length=50, fill='█', unfill='░', min_filled_length=1, print_percent=False)

# %%
class Problem():
    def __init__(self, environment: str = 'CartPole-v0', agent: Agent = RandomAgent, is_goal_reached: Callable[[np.array, gym.Env, float, bool], bool] = lambda state, environment, reward, done: False, output_filename : str = [], output_size : str = 'double', plot_results : bool = True, logger_level: int = logger.INFO):
        logger.set_level(logger_level)
        if len(output_filename) == 0:
            output_filename = 'results_' + environment + '.png'
        self.output_filename = output_filename
        self.output_size     = output_size
        self.plot_results    = plot_results
        self.environment     = gym.make(environment)
        self.agent           = agent(self.environment.observation_space,
                                     self.environment.action_space)
        self.is_goal_reached = is_goal_reached
        self.environment.seed(0)

    def __del__(self):
        try:
            self.environment.close()
        except:
            pass

    def _new_statistic_value_(self):
        return {'avg': [], 'min': [], 'max': []}

    def _print_details_(self):
        clear()
        if self.render:
            self.environment.render()
            print()
        print(f'Episode: {self.episode}, Reward: {self.reward}, Episode reward: {self.episode_reward}, Bach goals reached: {self.goals_reached} / {self.render_batch_size}')
        self.agent._print_details_()
        if self.done:
            time.sleep(10*(self.num_episodes - self.episode) / self.num_episodes)
        else:
            time.sleep(self.print_sleep_sec)

    def _print_progress_bars_(self):
        clear()
        print_progress_bar(self.render_batch, self.max_render_batches, prefix='Batches      ',
                           suffix=f'({self.render_batch} / {self.max_render_batches} total)', length=50)
        print_progress_bar(self.batch_episodes, self.render_batch_size, prefix='Episodes     ',
                           suffix=f'batch of {self.render_batch_size} ({self.episode} / {self.num_episodes} total)', length=50)
        print_progress_bar(self.goals_reached, self.render_batch_size, prefix='Goals reached',
                           suffix=f'batch of {self.render_batch_size} ({self.total_goals_reached} / {self.num_episodes} total)', length=50)
        self.agent._print_progress_bar_()

    def _plot_results_(self, x = [], y = [], xlim = [], ylim = [], xlabel = [], ylabel = [], color = []):
        def plot_value_statistics(ax, x, y, label = '', color = 'k', zorder = 1):
            ax.fill_between(x, y['min'], y['max'], linewidth=0, color=color, alpha=0.2, zorder=zorder)
            ax.plot(x, y['avg'], label=label, color=color, zorder=zorder)
        def plot(x, y, xlim = [], ylim = [], xlabel = '', ylabel = [], color = [], size = 'simple'):
            if len(ylabel) == 0:
                ylabel = [''] * len(y)
            if len(color) == 0:
                color = ['k'] * len(y)
            size = size.lower()
            lim  = lambda A: (min(A), max(A))
            if len(xlim) == 0:
                xlim = lim(x)
            if len(ylim) == 0:
                ylim = [xlim] + \
                       [lim(yi['avg'] + yi['min'] + yi['max'])
                            if isinstance(yi, dict) 
                            else lim(yi) for yi in y[1:]]
            plot_fnc = lambda ax, x, y, *args, **kwargs: \
                            plot_value_statistics(ax, x, y, *args, **kwargs) \
                            if isinstance(y, dict) else \
                            ax.plot(x, y, *args, **kwargs)
            # matplotlib.use("pgf")
            matplotlib.rcParams.update({
                'font.family'  : 'Times New Roman',
                'font.size'    : '8',
                'text.usetex'  : True,
                'pgf.texsystem': "pdflatex",
                'pgf.rcfonts'  : False
            })
            paper = {
                'width':  8.5,    # in
                'height': 11,     # in
                'colsep': 5 / 72, # in
                'margin': {
                    'left'  : 0.75, # in
                    'right' : 0.75, # in
                    'top'   : 1,    # in
                    'bottom': 1     # in
                }
            }
            content = {
                'width':  paper['width']  - paper['margin']['left'] - paper['margin']['right'] - paper['colsep'],
                'height': paper['height'] - paper['margin']['top']  - paper['margin']['bottom']
            }
            fig, ax0 = plt.subplots()
            if size == 'simple':
                fig.set_size_inches(w=content['width'] / 2, h=content['height'] / 4)
            elif size == 'double':
                fig.set_size_inches(w=content['width'] + paper['colsep'], h=content['height'] / 4)
            else:
                raise Exception('Figure size unknown')
            zorder = len(y)
            ax0.set_xlim(*xlim)
            ax0.set_ylim(*ylim[0])
            ax0.set_xlabel(xlabel)
            ax0.set_ylabel(ylabel[0], color=color[0])
            ax0.tick_params(axis='y', labelcolor=color[0])
            ax0.set_zorder(zorder)
            ax0.patch.set_visible(False)
            plt.setp(ax0.spines.values(), linewidth=0.5)
            plot_fnc(ax0, x, y[0], label=ylabel[0],  color=color[0])#, zorder=zorder)
            tight_bbox0  = ax0.get_tightbbox(fig.canvas.get_renderer())
            window_bbox0 = ax0.get_window_extent(fig.canvas.get_renderer())
            bbox0_label_width = (tight_bbox0.width - window_bbox0.width)
            for i in range(1, len(y)):
                zorder -= 1
                new_ax = ax0.twinx()
                new_ax.spines['right'].set_position(('outward', bbox0_label_width*(i-1)))
                new_ax.set_ylim(*ylim[i])
                new_ax.set_ylabel(ylabel[i], color=color[i])
                new_ax.tick_params(axis='y', labelcolor=color[i])
                new_ax.set_zorder(zorder)
                plt.setp(new_ax.spines.values(), linewidth=0.01)
                plot_fnc(new_ax, x, y[i], label=ylabel[i],  color=color[i])#, zorder=zorder)
            return fig
        if len(x) == 0:
            x =  self.statistics['episode']
        if len(y) == 0:
            y = (self.statistics['goals_reached'],
                 self.statistics['reward'])
        if len(xlabel) == 0:
            xlabel = 'Episode'
        if len(ylabel) == 0:
            ylabel = ('Cumulative goals reached',
                    'Rewards')
        if len(color) == 0:
            color = ('#0880AB',
                     '#CC4F1B')
        fig = plot(x, y, xlim, ylim, xlabel, ylabel, color, size=self.output_size)
        fig.tight_layout()
        fig.savefig(self.output_filename, dpi=300)

    def _reset_episode_(self):
        self.state = self.environment.reset()
        self.done = False
        self.trajectory = ()
        self.episode_reward = 0
        return self.state, self.done

    def _reset_run_(self, num_episodes = 1000, stat_batch_size = 100, render_batch_size = 100, render = True, print_details = False, plot_results = False, print_sleep_sec = 0, output_filename = [], output_size = []):
        self.num_episodes           = num_episodes
        self.stat_batch_size        = stat_batch_size
        self.render_batch_size      = render_batch_size
        self.render                 = render
        self.print_details          = print_details
        self.plot_results           = plot_results
        self.print_sleep_sec        = print_sleep_sec
        if len(output_filename) != 0:
            self.output_filename = output_filename
        if len(output_size) != 0:
            self.output_size = output_size
        self.rewards             = []
        self.total_goals_reached = 0
        self.goals_reached       = 0
        self.batch_goals_ratio   = 0
        self.render_batch        = 0
        self.batch_episodes      = 0
        self.max_render_batches  = math.ceil(num_episodes / render_batch_size)
        self.statistics          = {'episode': [], 'reward': self._new_statistic_value_(), 'goals_reached': []}

    def _save_step_(self, episode, state, action, next_state, reward, done, goal_reached):
        self.episode      = episode
        self.state        = state
        self.action       = action
        self.next_state   = next_state
        self.reward       = reward
        self.done         = done
        self.goal_reached = goal_reached

    def _terminate_episode_(self):
        self.rewards.append(self.episode_reward)
        self._update_statistics_batch_()
        self._update_render_batch_()
        if self.goal_reached:
            self.goals_reached += 1
            self.total_goals_reached += 1
        self.batch_episodes += 1
        self._print_progress_bars_()

    def _terminate_run_(self):
        self._show_step_()
        if self.plot_results:
            self._plot_results_()

    def _terminate_step_(self, episode, state, action, next_state, reward, done, goal_reached):
        self.trajectory     += ((state, action),)
        self.episode_reward += reward
        self._save_step_(episode, state, action, next_state, reward, done, goal_reached)
        self._show_step_()

    def _show_step_(self, force : bool = False):
        if force or self.episode % self.render_batch_size == 0:
            if self.print_details:
                self._print_details_()
            elif self.render:
                self.environment.render()

    def _update_statistics_batch_(self):
        self.statistics_bach_concluded = self.episode % self.stat_batch_size == 0
        if self.statistics_bach_concluded:
            batch_rewards = self.rewards[-self.stat_batch_size:]
            self.statistics['episode'].append(self.episode)
            self._update_value_statistics_('reward', batch_rewards)
            self.statistics['goals_reached'].append(self.total_goals_reached)

    def _update_render_batch_(self):
        self.render_bach_concluded = self.episode % self.render_batch_size == 0
        if self.render_bach_concluded:
            self.batch_goals_ratio = self.goals_reached / self.render_batch_size
            self.batch_episodes = 0
            self.goals_reached = 0
            self.render_batch += 1

    def _update_value_statistics_(self, name, values):
        self.statistics[name]['avg'].append(np.average(values))
        self.statistics[name]['min'].append(min(values))
        self.statistics[name]['max'].append(max(values))

    def run(self, num_episodes: int, stat_batch_size: int, render_batch_size: int, render: bool, print_details: bool, print_sleep_sec: float, output_filename: str, output_size: str) -> float:
        raise NotImplementedError

problems = {}

# %%
class RLProblem(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _plot_results_(self, *args, **kwargs):
        x =  self.statistics['episode']
        y = (self.statistics['goals_reached'],
             self.statistics['reward'],
             self.statistics['exploration_rate'])
        ylabel = ('Cumulative goals reached',
                  'Rewards',
                  'Exploration rate')
        color = ('#0880AB',
                 '#CC4F1B',
                 '#00A000')
        lim  = lambda A: (min(A), max(A))
        xlim = lim(x)
        ylim = [xlim] + \
               [lim(yi['avg'] + yi['min'] + yi['max'])
                   if isinstance(yi, dict) 
                   else lim(yi) for yi in y[1:-1]] + \
               [[0, 1]]
        super()._plot_results_(y=y, ylim=ylim, ylabel=ylabel, color=color, *args, **kwargs)

    def _reset_run_(self, *args, **kwargs):
        super()._reset_run_(*args, **kwargs)
        self.exploration_rate = []
        self.statistics['exploration_rate'] = self._new_statistic_value_()

    def _terminate_episode_(self, *args, **kwargs):
        self.exploration_rate.append(self.agent.exploration_rate)
        super()._terminate_episode_(*args, **kwargs)

    def _update_statistics_batch_(self, *args, **kwargs):
        super()._update_statistics_batch_(*args, **kwargs)
        if self.statistics_bach_concluded:
            batch_exploration_rate = self.exploration_rate[-self.stat_batch_size:]
            self._update_value_statistics_('exploration_rate', batch_exploration_rate)

    def run(self, *args, **kwargs) -> float:
        self._reset_run_(*args, **kwargs)
        for episode in range(self.num_episodes):
            state, done = self._reset_episode_()
            step = 0
            while not done:
                action = self.agent.step(state,episode)
                next_state, reward, done, _ = self.environment.step(action)
                goal_reached = self.is_goal_reached( next_state, self.environment, reward, done)
                self.agent.train(state, action, next_state, reward, done, goal_reached, self.batch_goals_ratio)
                self._terminate_step_(episode, state, action, next_state, reward, done, goal_reached)
                state = next_state
                step += 1
            self._terminate_episode_()
        self._terminate_run_()
        return self.episode_reward

# %%
# Source: https://github.com/the-computer-scientist/OpenAIGym/blob/7be80647e6e090c76f28ea03b7d1ba891db75f2f/QLearningIntro.ipynb
# Author: TheComputerScientist
# Date:   Oct 15, 2018
try:
    register(
        id                = 'FrozenLakeNoSlip-v0',
        entry_point       = 'gym.envs.toy_text:FrozenLakeEnv',
        kwargs            = {'map_name': '4x4', 'is_slippery': False},
        max_episode_steps = 100,
        reward_threshold  = 0.78
    )
except:
    pass

def is_goal_reached_FrozenLakeNoSlip_v0(state, environment, reward, done):
    return done and reward == 1

problems['FrozenLakeNoSlip-v0'] = RLProblem(environment="FrozenLakeNoSlip-v0", output_filename='doc/img/results_FrozenLakeNoSlip-v0.pgf', is_goal_reached=is_goal_reached_FrozenLakeNoSlip_v0,
                                            agent=lambda *args: QAgent(*args, discount_factor=0.99, learning_rate=0.85, exploration_decay_rate=0.99, min_exploration_rate=0.001))

# %%
def is_goal_reached_FrozenLake_v0(state, environment, reward, done):
    return done and reward == 1

problems['FrozenLake-v0'] = RLProblem(environment="FrozenLake-v0", output_filename='doc/img/results_FrozenLake-v0.pgf', is_goal_reached=is_goal_reached_FrozenLake_v0,
                                      agent=lambda *args: QAgent(*args, discount_factor=0.99, learning_rate=0.85, exploration_decay_rate=0.99, min_exploration_rate=0.001))

# %%
def is_goal_reached_MountainCar_v0(state, environment, reward, done):
    return state[0] >= environment.goal_position

problems['MountainCar-v0'] = RLProblem(environment="MountainCar-v0", output_filename='results_MountainCar-v0.png', is_goal_reached=is_goal_reached_MountainCar_v0,
                                       agent=lambda *args: QAgent(*args, discount_factor=0.99, learning_rate=0.9, exploration_decay_rate=0.99, min_exploration_rate=0.001, discretization_size=[20, None]))

# %%
def is_goal_reached_MountainCarContinuous_v0(state, environment, reward, done):
    return state[0] >= environment.goal_position

problems['MountainCarContinuous-v0'] = RLProblem(environment="MountainCarContinuous-v0", output_filename='results_MountainCarContinuous-v0.pgf', is_goal_reached=is_goal_reached_MountainCarContinuous_v0,
                                                 agent=lambda *args: QAgent(*args, discount_factor=0.99, learning_rate=0.9, exploration_decay_rate=0.99995, min_exploration_rate=0.001, discretization_size=[20, 20]))

# %%
# problem = problems['FrozenLakeNoSlip-v0']
problem = problems['FrozenLake-v0']
# problem = problems['MountainCar-v0']
# problem = problems['MountainCarContinuous-v0']

# problem.run(num_episodes=1,        render=False, render_batch_size=1,    print_details=False)
# problem.run(num_episodes=250,      render=False, render_batch_size=250,  print_details=False)
# problem.run(num_episodes=250*10,   render=False, render_batch_size=250,  print_details=False)
# problem.run(num_episodes=250*100,  render=True,  render_batch_size=250,  print_details=True, print_sleep_sec=0.5)
# problem.run(num_episodes=250*1000, render=False, render_batch_size=250,  print_details=False)

problem.run(num_episodes=250*100,  render=True,  render_batch_size=2500, print_details=True, print_sleep_sec=0.125)
# problem.run(num_episodes=250*100,  render=True,  render_batch_size=2500, print_details=False)

# %%
problem.run(num_episodes=1, render=True, render_batch_size=1, print_details=True, print_sleep_sec=0.125, plot_results=False)
# problem.run(num_episodes=1, render=True, render_batch_size=1, print_details=False, print_sleep_sec=0.125, plot_results=False)