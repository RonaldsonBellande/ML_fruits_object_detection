from header_imports import *

class model_building(models):
    def __init__(self, model_type, random_noise_count):

        self.image_file = []
        self.label_name = []
        self.random_noise_count = int(random_noise_count)
        self.image_size = 240
        self.number_of_nodes = 16
        self.valid_images = [".jpg",".png"]
        self.model = None
        self.model_summary = "model_summary/"
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model_type = model_type
        
        self.labelencoder = LabelEncoder()
        self.setup_structure() 
        self.splitting_data_normalize()
        
        if self.model_type == "model1":
            self.model = self.create_models_1()
        elif self.model_type == "model2":
            self.model = self.create_models_2()
        elif self.model_type == "model3":
            self.model = self.create_model_3()

        self.save_model_summary()

    
    def setup_structure(self):
        
        self.path  = "vehicle_image_data/"
        self.true_path = self.path
        self.category_names =  os.listdir(self.true_path)
        self.number_classes = len(next(os.walk(self.true_path))[1])
            
        for i in range(self.number_classes):
            self.check_valid(self.category_names[i])

        for i in range(self.number_classes):
            self.resize_image_and_label_image(self.category_names[i])

        self.label_name = self.labelencoder.fit_transform(self.label_name)
        self.image_file = np.array(self.image_file)
        self.label_name = np.array(self.label_name)
        self.label_name = self.label_name.reshape((len(self.image_file),1))

    
    def check_valid(self, input_file):
        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue
    

    def resize_image_and_label_image(self, input_file):
        for image in os.listdir(self.true_path + input_file):
            image_resized = cv2.imread(os.path.join(self.true_path + input_file,image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file.append(image_resized)
            self.label_name.append(input_file)
            # self.adding_random_noise(image_resized, input_file)


    def adding_random_noise(self, image, input_file):
        
        # Gaussian noise 
        for i in range(self.random_noise_count):
            gaussian_noise = np.random.normal(0, (10 **0.5), image.shape)
            image = image + gaussian_noise
            self.image_file.append(image)
            self.label_name.append(input_file)


        # Salt and pepper noise
        # This is too greedy
        # for i in range(self.random_noise_count):
        #     probability = 0.02
        #     for i in range(image.shape[0]):
        #         for j in range(image.shape[1]):
        #             random_num = random.random()
        #             if random_num < probability:
        #                 image[i][j] = 0
        #             elif random_num > (1 - probability):
        #                 image[i][j] = 255
        #     self.image_file.append(image)
        #     self.label_name.append(input_file)


        # Poisson noise
        for i in range(self.random_noise_count):
            poisson_noise = np.sqrt(image) * np.random.normal(0, 1, image.shape)
            noisy_image = image + poisson_noise
            self.image_file.append(image)
            self.label_name.append(input_file)


        # Speckle noise
        for i in range(self.random_noise_count):
            speckle_noise = np.random.normal(0, (10 **0.5), image.shape)
            image = image + image * speckle_noise
            self.image_file.append(image)
            self.label_name.append(input_file)


        # Uniform noise
        for i in range(self.random_noise_count):
            uniform_noise = np.random.uniform(0,(10 **0.5), image.shape)
            image = image + uniform_noise
            self.image_file.append(image)
            self.label_name.append(input_file)


    def splitting_data_normalize(self):
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size = 0.10, random_state = 42)
        self.input_shape = self.X_train.shape[1:]
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
        self.X_train = self.X_train.astype("float32") /255
        self.X_test = self.X_test.astype("float32") / 255

   
    
    def save_model_summary(self):
        with open(self.model_summary + self.model_type +"_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    



    
