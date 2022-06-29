from header_imports import *

@mark.usefixtures("class_fixture")
class TestFile(computer_vision_utilities):
    @fixture(autouse=True)
    def __init__(self):
        self.check_valid()


