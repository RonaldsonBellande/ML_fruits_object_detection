from header_imports import *

class segmentation(object):
    def __init__(self, saved_model, number_classes):
        
        self.image_file = []
        self.predicted_classes_array = []
        self.saved_model = saved_model
        self.model = keras.models.load_model("models/" + self.saved_model)
        self.image_path = "brain_cancer_category_2/" + "Testing2/" 

        self.image_size = 240
        self.number_classes = int(number_classes)
        self.split_size = 25
        self.color = [(0,255,255),(0,0,255),(0,255,0),(255,0,0)]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.alpha = 0.4
        self.fontScale = 0.5
        self.thickness = 1

        self.thickness_fill = -1
        self.graph_path = "graph_charts/" + "detection_localization/" 
        self.graph_path_localization = "graph_charts/" + "detection_localization/" + "localization/"
        self.graph_path_detection = "graph_charts/" + "detection_localization/" + "detection/"
        
        if self.number_classes == 2:
            self.model_categpory = ["False","True"]
            self.image_path = "brain_cancer_category_2/" + "Testing2/" 
       
        elif self.number_classes == 4:
            self.model_categpory = ["False", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
            self.image_path = "brain_cancer_category_4/" + "Testing2/" 

        self.prepare_image_data()
        self.plot_prediction_with_model()
        self.segmentation()

    
    def prepare_image_data(self):
        
        for image in os.listdir(self.image_path):
            image_resized = cv2.imread(os.path.join(self.image_path, image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.split_images(image_resized)
        
        self.image_file = np.array(self.image_file)
        self.X_test = self.image_file.astype("float32") / 255
    

    def split_images(self, image):

        for r in range(0,image.shape[0],int(math.sqrt(self.split_size))):
            for c in range(0,image.shape[1],int(math.sqrt(self.split_size))):
                image_split = image[r:r+int(math.sqrt(self.split_size)), c:c+int(math.sqrt(self.split_size)),:]
                image_split = cv2.resize(image_split,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
                self.image_file.append(image_split)


    def plot_prediction_with_model(self):

        predicted_classes = self.model.predict(self.X_test)

        for i in range(len(self.image_file)):
            if self.number_classes == 2:
                self.predicted_classes_array.append([np.argmax(predicted_classes[i])][0])
            elif self.number_classes == 4:
                self.predicted_classes_array.append([np.argmax(predicted_classes[i])][0])

        self.predicted_classes_array = np.reshape(self.predicted_classes_array, ((int(math.sqrt(len(self.image_file)))), (int(math.sqrt(len(self.image_file))))))
    

    def segmentation(self):

        first_prediction = False
        self.image_file_image_len = []
        for image in os.listdir(self.image_path):
            image_resized = cv2.imread(os.path.join(self.image_path, image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file_image_len.append(image_resized)
        
            image_resized_original = image_resized.copy()
        
            for j in range(len(self.image_file_image_len)):
                if self.number_classes == 2:
                    for i in range(len(self.model_categpory)):
                        for r in range(0,image_resized.shape[0],int(math.sqrt(self.split_size))):
                            for c in range(0,image_resized.shape[1],int(math.sqrt(self.split_size))):
                                if first_prediction == False:
                                    word_point = (int(r), int(c))
                                    first_prediction = True
                        
                                start_point = (int(r), int(c))
                                end_point = (int(r+(self.image_size/math.sqrt(len(self.image_file)))), int(c+(self.image_size/math.sqrt(len(self.image_file)))))
                                if self.predicted_classes_array[int(c/(self.image_size/(math.sqrt(len(self.image_file)))))][int(r/(self.image_size/(math.sqrt(len(self.image_file)))))] == i:
                                    image_resized=cv2.rectangle(image_resized, start_point, end_point, self.color[i], self.thickness_fill)
                                    image_resized=cv2.putText(image_resized, str((self.model_categpory[i])), word_point, self.font, self.fontScale, self.color[i], self.thickness, cv2.LINE_AA)

                    image_resized=cv2.addWeighted(image_resized, self.alpha, image_resized_original, (1-self.alpha), 3, image_resized_original)
                    cv2.imwrite(self.graph_path_detection + "model_segmenation_with_model_trained_prediction_" + str(self.saved_model)  + str(image) + str(j) + '.png', image_resized)
           
                if self.number_classes == 4:
                    for i in range(len(self.model_categpory)):
                        for r in range(0,image_resized.shape[0],int(math.sqrt(self.split_size))):
                            for c in range(0,image_resized.shape[1],int(math.sqrt(self.split_size))):
                                if first_prediction == False:
                                    word_point = (int(r), int(c))
                                    first_prediction = True
                        
                                start_point = (int(r), int(c))
                                end_point = (int(r+(self.image_size/math.sqrt(len(self.image_file)))), int(c+(self.image_size/math.sqrt(len(self.image_file)))))
                                if self.predicted_classes_array[int(c/(self.image_size/(math.sqrt(len(self.image_file)))))][int(r/(self.image_size/(math.sqrt(len(self.image_file)))))] == i:
                                    image_resized=cv2.rectangle(image_resized, start_point, end_point, self.color[i], self.thickness_fill)
                                    image_resized=cv2.putText(image_resized, str((self.model_categpory[i])), word_point, self.font, self.fontScale, self.color[i], self.thickness, cv2.LINE_AA)       
            

                    image_resized=cv2.addWeighted(image_resized, self.alpha, image_resized_original, (1-self.alpha), 3, image_resized_original)
                    cv2.imwrite(self.graph_path_detection + "model_segmenation_with_model_trained_prediction_" + str(self.saved_model) + str(image) + str(j) + '.png', image_resized)



