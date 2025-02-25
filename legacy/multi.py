import os
import pickle
import numpy as np
import multiprocessing as mp
import train_module  # train_module.py에 ExitStrategyEnv, QLearningAgent, train_agents 함수 등이 정의되어 있다고 가정

def run_training_instance(episodes):
    # 각 프로세스는 독립적으로 환경과 에이전트를 생성하여 학습합니다.
    agent1, agent2, _ = train_module.train_agents(episodes=episodes, agent1_epsilon=0.1, agent2_epsilon=0.1)
    # 두 에이전트의 Q-table을 하나로 합칩니다. (예: 단순히 두 테이블을 합치는 경우)
    return merge_q_tables([agent1.q_table, agent2.q_table])

def merge_q_tables(q_tables):
    """
    여러 Q-table을 병합하는 함수.
    동일한 키가 있을 경우, Q값들을 평균내어 사용합니다.
    """
    merged = {}
    counts = {}
    for q_table in q_tables:
        for key, value in q_table.items():
            if key in merged:
                merged[key] += value
                counts[key] += 1
            else:
                merged[key] = value
                counts[key] = 1
    for key in merged:
        merged[key] /= counts[key]
    return merged

def parallel_training(num_instances, episodes_per_instance):
    pool = mp.Pool(processes=num_instances)
    results = pool.map(run_training_instance, [episodes_per_instance] * num_instances)
    pool.close()
    pool.join()
    # results는 각 인스턴스에서 나온 병합된 Q-table의 리스트입니다.
    # 최종적으로 이 Q-table들을 다시 병합합니다.
    final_q_table = merge_q_tables(results)
    # 파일로 저장하거나 후속 학습에 활용할 수 있습니다.
    with open("merged_q_table.pkl", "wb") as f:
        pickle.dump(final_q_table, f)
    with open("agent1_q.pkl", "wb") as f:
        pickle.dump(final_q_table, f)
    with open("agent2_q.pkl", "wb") as f:
        pickle.dump(final_q_table, f)
    return final_q_table

if __name__ == "__main__":
    NUM_INSTANCES = 4          # 예: 4개의 병렬 프로세스
    EPISODES_PER_INSTANCE = 100  # 각 프로세스당 학습할 에피소드 수
    merged_q_table = parallel_training(NUM_INSTANCES, EPISODES_PER_INSTANCE)
    print("병렬 학습 완료, 병합된 Q-table의 크기:", len(merged_q_table))
