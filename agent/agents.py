def train_agents(episodes=1000):
    env = ExitStrategyEnv()
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            agent = agent1 if env.current_player == 1 else agent2
            action = agent.choose_action(env)
            if action is None:
                break
            reward, done = env.step(action)
            next_state = env.get_state()
            agent.update_q(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
    
    return agent1, agent2, rewards

def play_agents(agent1, agent2, episodes=10):
    results = {1: 0, 2: 0, 'draw': 0}

    for _ in range(episodes):
        env = ExitStrategyEnv()
        state = env.reset()
        done = False

        while not done:
            agent = agent1 if env.current_player == 1 else agent2
            action = agent.choose_action(env)
            if action is None:
                results['draw'] += 1
                break
            reward, done = env.step(action)
            state = env.get_state()
        
        if env.scores[1] == 2:
            results[1] += 1
        elif env.scores[2] == 2:
            results[2] += 1
        else:
            results['draw'] += 1

    return results

