from header_imports import *

@mark.usefixtures("class_fixture")
class TestModels(computer_vision_utilities):
    @fixture(autouse=True)
    def __init__(self):
        self.check_valid()
