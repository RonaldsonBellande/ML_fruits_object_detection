from header_imports import *

if __name__ == "__main__":
   
    with open("config/system_conf.json") as (file):
        system_config = json.load(file)

    if len(sys.argv) != 1:

        if sys.argv[2] == "model1":
            input_model = system_config["dataset"]["dataset_1_model1"]
        elif sys.argv[2] == "vit_transformer_shift_model":
            input_model = system_config["dataset"]["dataset_1_model1"]
        elif sys.argv[2] == "vit_transformer_shift_noise_model":
            input_model = system_config["dataset"]["dataset_1_model1"]
        elif sys.argv[2] == "unet_model":
            input_model = system_config["dataset"]["dataset_1_model1"]
        elif sys.argv[2] == "personal_model":
            input_model = system_config["dataset"]["dataset_1_model1"]
       

        if sys.argv[1] == "model_building":
            computer_vision_analysis_obj = model_building(config=system_config, model_type=sys.argv[2], random_noise_count=sys.argv[3])

        if sys.argv[1] == "model_training":
            computer_vision_analysis_obj = model_training(config=system_config, model_type=sys.argv[2], random_noise_count=sys.argv[3])

        if sys.argv[1] == "classification":
            computer_vision_analysis_obj = classification_with_model(config=system_config, saved_model=input_model)

        if sys.argv[1] == "transfer_learning":
            computer_vision_analysis_obj = transfer_learning(config=system_config, saved_model=input_model, model_type=sys.argv[2], random_noise_count=sys.argv[3])

        if sys.argv[1] == "continuous_learning":
            computer_vision_analysis_obj = continuous_learning(config=system_config, saved_model=input_model, model_type=sys.argv[2], episode=5, algorithm_name=sys.argv[4], transfer_learning="true")

        if sys.argv[1] == "classification_localization":
            computer_vision_analysis_obj = classification_localization(config=system_config, saved_model=input_model)

        if sys.argv[1] == "instance_segmentation":
            computer_vision_analysis_obj = instance_segmentation(config=system_config, saved_model=input_model)

        if sys.argv[1] == "semantic_segmentation":
            computer_vision_analysis_obj = semantic_segmentation(config=system_config, saved_model=input_model)



