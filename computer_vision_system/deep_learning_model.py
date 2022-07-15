from header_imports import *

class deep_q_learning(models):
    def __init__(self, saved_model, model_type, dense_size, batch_size, exploration_decay, algorithm_name, transfer_learning):

        self.delay_memory = 50000
        self.batch = batch_size
        self.dense_size = dense_size
        self.algorithm_name = algorithm_name
        self.q_learning_models = None
        self.gamma = 0.95
        self.target_update = 5
        self.exploration_decay = exploration_decay

        self.saved_model = saved_model
        self.model_type = model_type
        self.learning_rate = 0.001
        self.epochs = [1, 5, 15, 50, 100, 200]
        self.model_path = "models/transfer_learning/" + self.saved_model
        self.model_save_path =  "models/continuous_learning/" + self.saved_model
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999)
        self.transfer_learning = transfer_learning

        if self.transfer_learning == "true":
            self.model.load_weights(self.model_path)
        
        self.X_train = []
        self.Y_train = []
        
        if self.model_type == "model1":
            self.model = self.create_models_1()
        elif self.model_type == "vit_transformer_shift_model":
            self.model = self.vit_transformer_shift_model()
        elif self.model_type == "vit_transformer_shift_noise_model":
            self.model = self.vit_transformer_shift_noise_model()
        elif self.model_type == "unet_model":
            self.model = self.unet_model()
        elif self.model_type == "personal_model":
            self.model = self.personal_model()

        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen = self.delay_memory)

        self.callback_1 = TensorBoard(log_dir="logs/{}-{}".format(self.algorithm_name, int(time.time())))
        self.callback_2 = ModelCheckpoint(filepath=self.model_save_path, save_weights_only=True, verbose=1)
        self.callback_3 = ReduceLROnPlateau(monitor='accuracy', patience=2, verbose=1, factor= 0.5, min_lr=0.00001)
        self.target_update_counter = 0.001
        
        if self.algorithm_name != "deep_q_learning":
            self.update_target_model()


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


    def get_q_values(self):
        return self.model.predict(state)[0]


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
   

    def memory_delay(self):
        
        if self.transfer_learning == "true":
            self.model.load_weights(self.model_path)

        if len(self.replay_memory) > (self.batch):
            if self.algorithm_name == "deep_q_learning":
                self.train_deep_q_learning()
            elif self.algorithm_name == "double_deep_q_learning":
                self.train_double_deep_q_learning()
            elif self.algorithm_name == "dueling_deep_q_learning":
                self.train_dueling_deep_q_learning()
   

    def target_model_update(self, done):
        if done:
            self.target_update_counter += 1

        if self.target_update_counter > self.target_update:
            self.update_target_model()
            self.target_update_counter = 0
        

    def train_deep_q_learning(self):
        
        X = []
        Y = []
        
        self.mini_batch_sample = random.sample(self.replay_memory, self.batch)
        states_array = np.array([transition[0] for transition in self.mini_batch_sample]) 
        next_states_array = np.array([transition[3] for transition in self.mini_batch_sample])

        for index, (state, action, reward, next_state, done) in enumerate(self.mini_batch_sample):
            if done:
                state_value = reward
            else:
                state_value = reward + self.gamma *  np.max(self.model.predict(next_states_array)[index])
        
            state_value_action =  self.model.predict(states_array)[index]
            state_value_action[action] = state_value

            X.append(state)
            Y.append(state_value_action)

        self.X_train.extend(X)
        self.Y_train.extend(Y)
        
        self.q_learning_models = self.model.fit(np.array(X), np.array(Y), 
            batch_size=self.batch, 
            verbose=0, 
            epochs=self.epochs[0], 
            shuffle=False, 
            callbacks=[self.callback_1, self.callback_2, self.callback_3] if done else None)

        self.transfer_learning = self.train_initial_model


    def train_double_deep_q_learning(self):
        
        X = []
        Y = []

        self.mini_batch_sample = random.sample(self.replay_memory, self.batch)
        states_array = np.array([transition[0] for transition in self.mini_batch_sample]) 
        next_states_array = np.array([transition[3] for transition in self.mini_batch_sample])

        for index, (state, action, reward, next_state, done) in enumerate(self.mini_batch_sample):
            if done:
                state_value = reward
            else:
                state_value = reward + self.gamma *  np.max(self.target_model.predict(next_states_array)[index])
        
            state_value_action =  self.model.predict(states_array)[index]
            state_value_action[action] = state_value

            X.append(state)
            Y.append(state_value_action)
        
        self.X_train.extend(X)
        self.Y_train.extend(Y)
        
        self.q_learning_models = self.model.fit(np.array(X), np.array(Y), 
            batch_size=self.batch, 
            verbose=0, 
            epochs=self.epochs[0], 
            shuffle=False, 
            callbacks=[self.callback_1, self.callback_2, self.callback_3] if done else None)

        self.transfer_learning = self.train_initial_model


    def train_dueling_deep_q_learning(self):
        
        X = []
        Y = []
        
        self.mini_batch_sample = random.sample(self.replay_memory, self.batch)
        current_states = np.array([transition[0] for transition in self.mini_batch_sample]) 
        new_current_states = np.array([transition[3] for transition in self.mini_batch_sample])

        for index, (state, action, reward, next_state, done) in enumerate(self.mini_batch_sample):
            if done:
                state_value = reward
            else:
                state_value = reward + self.gamma *  np.max(self.target_model.predict(new_current_states)[index])
        
            state_value_action =  self.model.predict(states_array)[index]
            state_value_action[action] = state_value

            X.append(state)
            Y.append(state_value_action)
       

        self.X_train.extend(X)
        self.Y_train.extend(Y)
        
        self.q_learning_models = self.model.fit(np.array(X), np.array(Y), 
            batch_size=self.batch, 
            verbose=0, 
            epochs=self.epochs[0], 
            shuffle=False, 
            callbacks=[self.callback_1, self.callback_2, self.callback_3] if done else None)

        self.transfer_learning = self.train_initial_model


    def save_model(self):
        self.model.save(self.model_save_path)
