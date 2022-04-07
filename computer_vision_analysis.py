from header_imports import *

if __name__ == "__main__":
    
    if len(sys.argv) != 1:
        if sys.argv[1] == "create":
            util = utilities()
            util.seperate_image_base_on_image(nested_folders = "True")

        if sys.argv[1] == "seperate":
            util = utilities()
            util.seperate_image_into_file()

        if sys.argv[1] == "model_building":
            brain_analysis_obj = model_building(model_type=sys.argv[2], random_noise_count=sys.argv[3])

        if sys.argv[1] == "model_training":
            brain_analysis_obj = model_training(model_type=sys.argv[2], random_noise_count=sys.argv[3])

        if sys.argv[1] == "image_prediction":
             
            if sys.argv[2] == "model1":
                input_model = "model1_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model2":
                input_model = "model2_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model3":
                input_model = "model3_computer_vision_categories_2_model.h5"

            computer_vision_analysis_obj = classification_with_model(saved_model=input_model)

        if sys.argv[1] == "transfer_learning":

            if sys.argv[2] == "model1":
                input_model = "model1_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model2":
                input_model = "model2_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model3":
                input_model = "model3_computer_vision_categories_2_model.h5"

            computer_vision_analysis_obj = transfer_learning(saved_model=input_model, model_type=sys.argv[2], random_noise_count=sys.argv[3])

        if sys.argv[1] == "continuous_learning":
            
            if sys.argv[2] == "model1":
                input_model = "model1_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model2":
                input_model = "model2_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model3":
                input_model = "model3_computer_vision_categories_2_model.h5"
            
            computer_vision_analysis_obj = continuous_learning(saved_model=input_model, model_type=sys.argv[2], episode=5, algorithm_name=sys.argv[4], transfer_learning="true")

        if sys.argv[1] == "localization_detection":
            
            if sys.argv[2] == "model1":
                input_model = "model1_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model2":
                input_model = "model2_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model3":
                input_model = "model3_computer_vision_categories_2_model.h5"

            computer_vision_analysis_obj = localization_detection(saved_model=input_model)


        if sys.argv[1] == "segmentation":
            
            if sys.argv[2] == "model1":
                input_model = "model1_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model2":
                input_model = "model2_computer_vision_categories_2_model.h5"
            elif sys.argv[2] == "model3":
                input_model = "model3_computer_vision_categories_2_model.h5"

            computer_vision_analysis_obj = segmentation(saved_model=input_model)



