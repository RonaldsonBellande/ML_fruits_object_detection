from header_imports import *


class plot_graphs(object):
    def __init__(self):

        self.true_path = self.path + "Testing/"
        self.number_images_to_plot = 1


    def plot_episode_time_step(self, data, type_graph):

        fig = plt.figure()
        axis = fig.add_subplot(111)
        color_graph = "blue"

        if type_graph == "cumulative_reward":
            axis.plot(data, color=color_graph)
            axis.set_title("Reward vs Episode")
            axis.set_xlabel("Episode")
            axis.set_ylabel("Reward per Step")
        elif type_graph == "step_number":
            axis.plot(data, color=color_graph)
            axis.set_title("Number of steps per episode vs. Episode")
            axis.set_xlabel("Episode")
            axis.set_ylabel("step per episode")
        plt.savefig(self.algorithm_details + self.algorithm_name  + "_" + self.model_type + "_" + type_graph + ".png", dpi =500)


    def plot_model(self):

        plt.plot(self.q_learning_models.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.model_detail + self.algorithm_name + self.model_type + '_accuracy' + '.png', dpi =500)
        plt.clf()

        plt.plot(self.q_learning_models.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig(self.model_detail + self.algorithm_name + self.model_type + '_lost'+'.png', dpi =500)
        plt.clf()


    def plot_prediction_with_model(self):

        plt.figure(dpi=500)
        predicted_classes = self.model.predict(self.X_test)
        
        print(predicted_classes)
        for i in range(self.number_images_to_plot):
            fig=plt.imshow(self.X_test[i,:,:,:])
            plt.subplot(1,1,i+1)
            plt.axis('off')
            plt.title("Predicted - {}".format(self.category_names[np.argmax(predicted_classes[i], axis=0)]), fontsize=1)
            plt.tight_layout()
            plt.savefig(self.graph_path + "model_classification_detection_with_model_trained_prediction_continuous_learning" + str(self.saved_model) + '.png')

