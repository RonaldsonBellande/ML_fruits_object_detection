from header_imports import *

@mark.usefixtures("class_fixture")
class TestPaths(object):
    @fixture(autouse=True)
    def method_fixture(self):
