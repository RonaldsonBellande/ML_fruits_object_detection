from header_imports import *

class computer_vision_utilities(object):
    def setup_structure(self):
        
        self.path  = "fruits_360_datasets/"
        self.true_path = self.path + "Training_Small/"
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
        for i in range(self.random_noise_count):
            probability = 0.02
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    random_num = random.random()
                    if random_num < probability:
                        image[i][j] = 0
                    elif random_num > (1 - probability):
                        image[i][j] = 255
            self.image_file.append(image)
            self.label_name.append(input_file)


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
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size=0.10, random_state=42)
        self.input_shape = self.X_train.shape[1:]
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
        self.X_train = self.X_train.astype("float32") /255
        self.X_test = self.X_test.astype("float32") /255




class utilities(object):
    def __init__(self, name_of_new_directory = "brain_cancer_seperate/"):
        self.path = "/Data2"
        self.seperate_path = "Data2/Data"
        self.file_path_to_move = "brain_cancer_seperate/"
        self.valid_images = [".jpg",".png"]
        self.name_of_new_directory = name_of_new_directory


    def seperate_image_base_on_image(self, nested_folders = "None", directory_name = "True - False"):
        
        directory_array = directory_name.split(" - ")
        if nested_folders == "None":
            if os.path.isdir(self.name_of_new_directory) == False:
                os.mkdir(self.name_of_new_directory)
        else:
            for i in range(len(directory_array)):
                if os.path.isdir(str(self.name_of_new_directory + directory_array[i])) == False:
                    os.mkdir(str(self.name_of_new_directory + directory_array[i]))


    def seperate_image_into_file(self):
        list_images = os.listdir(self.seperate_path)
        for image in list_images:
            if image.endswith(self.valid_images[0]) or image.endswith(self.valid_images[1]):
                if 'y' in image.lower():
                    shutil.copy(os.path.join(self.seperate_path, image), self.file_path_to_move + "True")
                elif 'n' in image.lower():
                    shutil.copy(os.path.join(self.seperate_path, image), self.file_path_to_move + "False")
                else:
                    print("error")
