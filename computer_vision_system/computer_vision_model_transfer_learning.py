from header_imports import *


class transfer_learning(models, computer_vision_utilities):
    def __init__(self, saved_model, model_type, random_noise_count):
        
        self.image_file = []
        self.label_name = []
        self.random_noise_count = int(random_noise_count)
        self.image_size = 240
        self.saved_model = saved_model

        self.valid_images =  self.config["dataset"]["valid_images"]
        self.model_summary =  self.config["building"]["model_summary"]
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model_type = model_type
        
        self.labelencoder = LabelEncoder()
        self.setup_structure() 
        self.splitting_data_normalize()
        
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

        self.model.load_weights("models/" + self.saved_model)

        self.number_images_to_plot = self.config["transfer_learning"]["number_images_to_plot"] 
        self.batch_size = self.config["model"]["batch_size"] 
        self.epochs = self.config["model"]["epochs"] 

        self.graph_path = self.config["transfer_learning"]["graph_path"] 
        self.model_path = self.config["transfer_learning"]["model_path"]

        self.param_grid = dict(batch_size=self.batch_size, epochs=self.epochs)
        self.callback_1 = TensorBoard(log_dir="logs/{}-{}".format(self.model_type, int(time.time())))
        self.callback_2 = ModelCheckpoint(filepath=self.model_path, save_weights_only=True, verbose=1)
        self.callback_3 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor= 0.5, min_lr=0.00001)
        
        self.train_model()
        self.evaluate_model()
        self.plot_model()
        self.plot_prediction_with_model()



    def train_model(self):
       
        grid = GridSearchCV(estimator = self.model, param_grid = self.param_grid, n_jobs = 1, cv = 3, verbose = 10)

        self.computer_vision_model = self.model.fit(self.X_train, self.Y_train,
                batch_size=self.batch_size[0],
                validation_split=0.15,
                epochs=self.epochs[3],
                callbacks=[self.callback_1, self.callback_2, self.callback_3],
                shuffle=True)

        self.model.save(self.model_path + self.model_type + "_computer_vision_categories_"+ str(self.number_classes)+"_model.h5")
   

    def evaluate_model(self):
        evaluation = self.model.evaluate(self.X_test, self.Y_test, verbose=1)

        with open(self.graph_path + self.model_type + "_evaluate_computer_vision_category_" + str(self.number_classes) + ".txt", 'w') as write:
            write.writelines("Loss: " + str(evaluation[0]) + "\n")
            write.writelines("Accuracy: " + str(evaluation[1]))
        
        print("Loss:", evaluation[0])
        print("Accuracy: ", evaluation[1])


    def plot_model(self):

        plt.plot(self.computer_vision_model.history['accuracy'])
        plt.plot(self.computer_vision_model.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.graph_path + self.model_type + '_accuracy_' + str(self.number_classes) + '.png', dpi =500)
        plt.clf()

        plt.plot(self.computer_vision_model.history['loss'])
        plt.plot(self.computer_vision_model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.graph_path + self.model_type + '_lost_' + str(self.number_classes) +'.png', dpi =500)
        plt.clf()


    def plot_prediction_with_model(self):

        plt.figure(dpi=500)
        predicted_classes = self.model.predict(self.X_test)
        
        for i in range(self.number_images_to_plot):
            plt.subplot(4,4,i+1)
            fig=plt.imshow(self.X_test[i,:,:,:])
            plt.axis('off')
            plt.title("Predicted - {}".format(self.category_names[np.argmax(predicted_classes[i], axis=0)]) + "\n Actual - {}".format(self.category_names[np.argmax(self.Y_test_vec[i,0])]),fontsize=1)
            plt.tight_layout()
            plt.savefig(self.graph_path + "model_detection_localization_with_model_trained_prediction_" + str(self.saved_model) + '.png')

        
