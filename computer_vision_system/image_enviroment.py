from header_imports import *


class classification_enviroment(gym.Env):
    def __init__(self, number_classes, data_set, image_per_episode):

        self.number_classes = number_classes
        self.images_per_episode = image_per_episode
        self.step_count = 0
        self.X, self.Y = data_set[0], data_set[1]
        self.action_space = spaces.Discrete(self.number_classes)
        self.state_space = spaces.Box(low=0, high=1, shape=(self.X.shape[1], self.X.shape[2], 1), dtype=np.float32)


    def step(self, action):
        done = False
        action = np.argmax(action.numpy())
        reward = int(action == self.expected_action)
        next_state = self.state()
        self.step_count += 1
        if self.step_count >= self.images_per_episode:
            done = True
        
        return action, reward, next_state, done
    

    def state(self):
        next_state_idx = random.randint(0, len(self.X) - 1)
        self.expected_action = np.argmax(self.Y[next_state_idx])
        state_space = self.X[next_state_idx]
        return state_space


    def reset(self):
        self.step_count = 0
        next_state = self.state()
        return next_state


