from loguru import logger
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('Taxi-v3', render_mode='human')
observation, info = env.reset()
print(f'OBSERVATION SPACE:\n\n{observation}\n')
print(f'ACTION SPACE SPECS:\n\n{info}')

episode_history = []
reward_history = []
iteration = 0
while True:
    iteration += 1
    action = env.action_space.sample()  # agent policy that uses the observation and info
    ret_state = env.step(action)
    episode = len(episode_history) + 1
    logger.info(f'{episode=} : {iteration} : {ret_state}')
    observation, reward, terminated, truncated, info = ret_state
    reward_history.append(reward)

    if terminated or truncated:
        terminated_state = env.reset()
        observation, info = terminated_state
        reward_cumsum = np.cumsum(reward_history)
        episode_history.append(reward_cumsum)
        reward_history = []
        plt.plot(range(1, len(reward_cumsum)+1), reward_cumsum)
        plt.show()
        logger.success(f'END OF EPISODE: {iteration} : {terminated_state}')
        iteration = 0


        if len(episode_history) == 10:
            # plot average performance
            historical_performance = [rh[-1] for rh in episode_history]
            hist_perf_cumsum = np.cumsum(historical_performance)
            hist_perf_cumsum_mean = [cs / (idx + 1) for idx, cs in enumerate(hist_perf_cumsum)]
            plt.plot(range(1, len(hist_perf_cumsum_mean) + 1), hist_perf_cumsum_mean)
            plt.show()
            break

logger.success(f'Simulation Completed after {len(episode_history)} episode.')
env.close()
pass