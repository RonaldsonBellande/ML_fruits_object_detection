from header_imports import *

class prediction_with_model(object):
    def __init__(self, model, image_path):

        self.image_path = image_path
        self.model = model

        img = cv2.imread(self.image_path)
        img = cv2.resize(img, (224, 224))
        
        index = self.predict_image(img)


    def predict_image(self, image):

        dims = np.expand_dims(image, axis=0)
        dims = preprocess_input(dims)
        class_probabilities = self.model.predict(dims)
        index = int(np.argmax(class_probabilities, axis=1)[0])

        return index
