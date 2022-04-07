from header_import import *


class deep_q_learning_algorithm(DeepQLearning, plot_graphs):
    def __init__(self, episode, noise=0.0, reward_noise=0.0, random_start=False, state_world_size=400, algorithm_name="deep_q_learning", transfer_learning="true", image_base="false"):
        super().__init__(algorithm_name=algorithm_name, transfer_learning=transfer_learning)
        
        self.path = "graph_charts/"
        self.enviroment_path = self.path + "enviroment_details/"
        self.model_detail_path = self.path + "model_details/"
        
        self.train_initial_model = "false"
        self.algorithm_name = algorithm_name
        self.transfer_learning = transfer_learning
        self.episode = episode
        self.step_limit = 500
        self.epsilon = 1
        self.delay_epsilon = 0.995
        self.min_epsilon = 0.001
        self.low_state_bound = np.array([-1.2, -0.07])
        self.high_state_bound = np.array([0.6, 0.07])
        self.normalize = np.subtract(self.high_state_bound, self.low_state_bound)
        self.state_size = (4,)
        self.action_size = 5
        self.state_world_size = state_world_size
        self.position_range = np.linspace(self.low_state_bound[0], self.high_state_bound[0], self.state_world_size)
        self.velocity_range = np.linspace(self.low_state_bound[1], self.high_state_bound[1], int(self.state_world_size/5))
        self.episode_rewards = []
        self.step_per_episode = []


    def policy(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.get_q_values(state))
        else:
            return np.random.randint(0, self.action_size)


    def epsilon_reduction(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.delay_epsilon


    def deep_q_learning(self):
    
        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state = self.reset()
            self.reach_goal = False
            episode_reward = 0

            while not self.reach_goal and step <= self.step_limit:
                action = self.policy(state)
                action, reward, next_state, self.reach_goal = self.step(action)
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, self.reach_goal))
                state = next_state
                self.memory_delay()
                step += 1
            
            self.train_initial_model = "true"
            self.epsilon_reduction()
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        self.plot_episode_time_step(self.episode_rewards,type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph="step_number")
        self.plot_model()


    def double_deep_q_learning(self):

        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state = self.reset()
            self.reach_goal = False
            episode_reward = 0

            while not self.reach_goal and step <= self.step_limit:
                action = self.policy(state)
                action, reward, next_state, self.reach_goal = self.step(action)
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, self.reach_goal))
                state = next_state
                self.target_model_update()
                self.memory_delay()
                step += 1
                
            self.train_initial_model = "true"
            self.epsilon_reduction()
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        self.plot_episode_time_step(self.episode_rewards,type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph = "step_number")
        self.plot_model()


    def dueling_deep_q_learning(self):

        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state = self.reset()
            self.reach_goal = False
            episode_reward = 0

            while not self.reach_goal and step <= self.step_limit:
                action = self.policy(state)
                action, reward, next_state, self.reach_goal = self.step(action)
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, self.reach_goal))
                state = next_state
                self.target_model_update()
                self.memory_delay()
                step += 1

            self.train_initial_model = "true"
            self.epsilon_reduction()
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        self.plot_episode_time_step(self.episode_rewards, type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph = "step_number")
        self.plot_model()

