from header_imports import *

class model_building(models, computer_vision_utilities, model_utilities):
    def __init__(self, config, model_type, random_noise_count):

        model_utilities.__init__(self)
        
        self.config = config
        self.image_file = []
        self.label_name = []
        self.random_noise_count = int(random_noise_count)
        self.valid_images = [".jpg",".png"]
        self.model = None
        self.model_summary = self.config["building"]["model_summary"]
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model_type = model_type
        
        self.labelencoder = LabelEncoder()
        self.setup_structure()
        self.splitting_data_normalize()

        if self.model_type == "model1":
            self.model = self.create_models_1()
        elif self.model_type == "vit_transformer_model":
            self.model = self.vit_transformer_model()
        elif self.model_type == "unet_model":
            self.model = self.unet_model()
        elif self.model_type == "personal_model":
            self.model = self.personal_model()

        self.save_model_summary()
        self.display_model_archetecture()



    def save_model_summary(self):
        with open(self.model_summary + self.model_type +"_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    def display_model_archetecture(self):
        keras.utils.plot_model(self.model, to_file=self.model_summary + self.model_type +"_architecture_" + str(self.number_classes) +".png", show_shapes=True, show_layer_names=True)


    



    
