from header_imports import *

class utilities(object):

    def __init__(self, name_of_new_directory = "brain_cancer_seperate/"):
        self.path = "/Data2"
        self.seperate_path = "Data2/Data"
        self.file_path_to_move = "brain_cancer_seperate/"
        self.valid_images = [".jpg",".png"]
        self.name_of_new_directory = name_of_new_directory


    def seperate_image_base_on_image(self, nested_folders = "None", directory_name = "True - False"):
        
        directory_array = directory_name.split(" - ")
        if nested_folders == "None":
            if os.path.isdir(self.name_of_new_directory) == False:
                os.mkdir(self.name_of_new_directory)
        else:
            for i in range(len(directory_array)):
                if os.path.isdir(str(self.name_of_new_directory + directory_array[i])) == False:
                    os.mkdir(str(self.name_of_new_directory + directory_array[i]))


    def seperate_image_into_file(self):
        list_images = os.listdir(self.seperate_path)
        for image in list_images:
            if image.endswith(self.valid_images[0]) or image.endswith(self.valid_images[1]):
                if 'y' in image.lower():
                    shutil.copy(os.path.join(self.seperate_path, image), self.file_path_to_move + "True")
                elif 'n' in image.lower():
                    shutil.copy(os.path.join(self.seperate_path, image), self.file_path_to_move + "False")
                else:
                    print("error")
