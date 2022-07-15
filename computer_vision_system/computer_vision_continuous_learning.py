from header_imports import *


class continuous_learning(deep_q_learning, classification_enviroment, plot_graphs, computer_vision_utilities):
    def __init__(self, config, saved_model, model_type, number_classes, image_type, episode, noise=0.0, reward_noise=0.0, state_world_size=400, algorithm_name="deep_q_learning", transfer_learning="true"):
        
        self.config = config
        self.algorithm_details = self.config["continuous_learning"]["algorithm_details"] 
        self.model_detail =  self.config["continuous_learning"]["model_detail"] 
        self.graph_path =  self.config["continuous_learning"]["graph_path"] 
       
        self.number_classes = int(number_classes)
        self.image_type = image_type
        self.image_size = 240
        self.dense_size = 10
        self.exploration_decay = 0.95
        self.image_file = []
        self.label_name = []
        self.saved_model = saved_model
        self.model_type = model_type
        self.batch_size = self.config["model"]["batch_size"] 
        self.epochs = self.config["model"]["epochs"] 

        self.graph_path = self.config["continuous_learning"]["graph_path"] 
        self.model_path = self.config["continuous_learning"]["model_path"]

        self.number_of_points = 2048
        self.number_images_to_plot = self.config["transfer_learning"]["number_images_to_plot"] 
        self.valid_images =  self.config["dataset"]["valid_images"]
        self.labelencoder = LabelEncoder()
        
        self.setup_structure()
        self.splitting_data_normalize()

        self.image_per_episode = int(math.sqrt(len(self.image_file)))
        self.train_initial_model = "false"
        self.algorithm_name = algorithm_name
        self.transfer_learning = transfer_learning
        self.episode = episode
        self.step_limit = 10
        self.epsilon = 1
        self.delay_epsilon = 0.995
        self.min_epsilon = 0.001
        self.episode_rewards = []
        self.step_per_episode = []

        deep_q_learning.__init__(self, config=self.config , saved_model=self.saved_model, model_type=self.model_type, dense_size=self.dense_size, batch_size=self.batch_size[3], exploration_decay=self.exploration_decay, algorithm_name=self.algorithm_name, transfer_learning=self.transfer_learning)
        classification_enviroment.__init__(self, config=self.config , number_classes=self.number_classes, data_set=(self.image_file, self.label_name), image_per_episode=self.image_per_episode)
        
        if self.algorithm_name == "deep_q_learning":
            self.deep_q_learning()
        elif self.algorithm_name == "double_deep_q_learning":
            self.double_deep_q_learning()
        elif self.algorithm_name == "dueling_deep_q_learning":
            self.dueling_deep_q_learning()

        plot_graphs.__init__(self)


    def deep_q_learning(self):
    
        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state, done = self.reset(), False
            episode_reward = 0

            for i in tqdm(range(1, self.image_per_episode), desc="image_per_episode"):
                action, reward, next_state, done = self.step(self.model(state[None])[0])
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, done))
                state = next_state
                self.memory_delay()
                step += 1
            
            self.train_initial_model = "true"
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        self.plot_episode_time_step(self.episode_rewards,type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph="step_number")
        # self.plot_model()
        self.plot_prediction_with_model()


    def double_deep_q_learning(self):

        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state, done = self.reset(), False
            episode_reward = 0

            for i in tqdm(range(1, self.image_per_episode), desc="image_per_episode"):
                action, reward, next_state, done = self.step(self.model(state[None])[0])
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, done))
                state = next_state
                self.target_model_update(done)
                self.memory_delay()
                step += 1
                
            self.train_initial_model = "true"
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        self.plot_episode_time_step(self.episode_rewards,type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph = "step_number")
        # self.plot_model()
        self.plot_prediction_with_model()


    def dueling_deep_q_learning(self):

        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state, done = self.reset(), False
            episode_reward = 0
            
            for i in tqdm(range(1, self.image_per_episode), desc="image_per_episode"):
                action, reward, next_state, done = self.step(self.model(state[None])[0])
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, done))
                state = next_state
                self.target_model_update(done)
                self.memory_delay()
                step += 1

            self.train_initial_model = "true"
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        self.plot_episode_time_step(self.episode_rewards, type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph = "step_number")
        # self.plot_model()
        self.plot_prediction_with_model()


