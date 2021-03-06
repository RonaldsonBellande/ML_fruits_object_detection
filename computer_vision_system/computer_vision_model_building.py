from header_imports import *

class model_building(models, computer_vision_utilities, model_utilities, freesing_model):
    def __init__(self, config, model_type, random_noise_count):
        model_utilities.__init__(self)
        
        self.config = config
        self.image_file = []
        self.label_name = []
        self.random_noise_count = int(random_noise_count)
        self.valid_images = self.config["dataset"]["valid_images"]
        self.model = None

        self.model_summary = self.config["building"]["model_summary"]
        self.model_parameters = self.model_summary + self.config["building"]["model_parameters"]
        self.model_image = self.model_summary + self.config["building"]["model_image"]
        self.model_h5 = self.model_summary + self.config["building"]["model_h5"]
        self.model_pb = self.model_summary + self.config["building"]["model_pb"]

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
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
        
        self.save_model_summary()
        self.display_model_archetecture()
        # self.model_archetecture_h5()
        self.h5_to_pb()
        self.model_archetecture_onnx()



    def save_model_summary(self):
        with open(self.model_parameters + self.model_type + "_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    def display_model_archetecture(self):
        keras.utils.plot_model(self.model, to_file=self.model_image + self.model_type + "_architecture_" + str(self.number_classes) +".png", show_shapes=True, show_layer_names=True)


    def model_archetecture_h5(self):
        self.model.save(self.model_h5 + self.model_type + "_h5_architecture_"+ str(self.number_classes)+"_model.h5")


    def h5_to_pb(self):
        output_names = [out.op.name for out in self.model.outputs]
        
        frozen_graph = self.freeze(output_names)
        tf1.train.write_graph(frozen_graph, self.model_pb , self.model_type + "_architecture_pd_" + str(self.number_classes) +".pd", as_text=False)


    def model_archetecture_onnx(self):
        pass

