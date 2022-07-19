from header_imports import *

class computer_vision_utilities(model_utilities):
    def setup_structure(self):
        
        self.true_path = self.config["dataset"]["dataset_path"]
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


    def splitting_data_normalize(self):
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size=0.10, random_state=42)
        self.input_shape = self.X_train.shape[1:]
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
        self.X_train = self.X_train.astype("float32") /255
        self.X_test = self.X_test.astype("float32") /255


class freesing_model(object):
    def freeze(keep_var_names=None, output_names=None, clear_devices=True):
        
        graph = K.get_session().graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph
