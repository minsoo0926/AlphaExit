from ExitStrategyEnv import ExitStrategyEnv
from QLearningAgent import QLearningAgent
import pickle

def train_agents(episodes=1000):
    env = ExitStrategyEnv()
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()
    rewards = []
    
    for episode in range(episodes):
        env.reset()
        done = False
        total_reward = 0

        while not done:
            agent = agent1 if env.current_player == 1 else agent2
            action = agent.choose_action(env)
            if action is None:
                break
            reward, done = env.step(action)
            next_state = env  # Pass the environment object
            agent.update_q(env, action, reward, next_state)  # Pass the environment object
            total_reward += reward
        
        rewards.append(total_reward)
    
    return agent1, agent2, rewards

def play_agents(agent1, agent2, episodes=10):
    results = {1: 0, 2: 0, 'draw': 0}

    for _ in range(episodes):
        env = ExitStrategyEnv()
        env.reset()
        done = False

        while not done:
            agent = agent1 if env.current_player == 1 else agent2
            action = agent.choose_action(env)
            if action is None:
                results['draw'] += 1
                break
            reward, done = env.step(action)
            env = env  # Pass the environment object
        
        if env.scores[1] == 2:
            results[1] += 1
        elif env.scores[2] == 2:
            results[2] += 1
        else:
            results['draw'] += 1

    return results

def save_q_table(agent, filename="q_table.pkl"):
    """학습된 Q-table 저장"""
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)

def load_q_table(filename="q_table.pkl"):
    """저장된 Q-table 불러오기"""
    with open(filename, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    agent1, agent2, _ = train_agents()
    save_q_table(agent1, "q_table_1.pkl")
    save_q_table(agent2, "q_table_2.pkl")