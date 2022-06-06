from header_imports import *

class model_building(models, computer_vision_utilities):
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
        
        # Transformer
        self.patch_size = 6  # Size of the patches to be extract from the input images
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [self.projection_dim * 2, self.projection_dim]  # Size of the transformer layers
        self.transformer_layers = 8
        self.mlp_head_units = [2048, 1024]
        self.epsilon = 1e-6

        if self.model_type == "model1":
            self.model = self.create_models_1()
        elif self.model_type == "model2":
            self.model = self.vit_transformer_model_1()
        elif self.model_type == "model3":
            self.model = self.create_model_3()

        self.save_model_summary()


    def save_model_summary(self):
        with open(self.model_summary + self.model_type +"_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    



    
