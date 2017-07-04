from vtools import Tree, _WIDTH, _DEPTH, imgPaths
import os

def test_tree():

    tree = Tree()
    tree.create_node("Harry", "harry")  # root node
    tree.create_node("Jane", "jane", parent = "harry")
    tree.create_node("Bill", "bill", parent = "harry")
    tree.create_node("Joe", "joe", parent = "jane")
    tree.create_node("Diane", "diane", parent = "jane")
    tree.create_node("George", "george", parent = "diane")
    tree.create_node("Mary", "mary", parent = "diane")
    tree.create_node("Jill", "jill", parent = "george")
    tree.create_node("Carol", "carol", parent = "jill")
    tree.create_node("Grace", "grace", parent = "bill")
    tree.create_node("Mark", "mark", parent = "jane")

    print("="*80)
    tree.show("harry")
    print("="*80)
    for node in tree.expand_tree("harry", mode=_WIDTH):
        print(node)
    print("="*80)

class vtools_testSuite:
    """ Class for vtools testSuite. Broke testing units into classes because it made
        sense


    """
    def __init__(self):
        self.tests = (self.test_imgPaths,)

    def __call__(self):
        for test in self.tests:
            test()

    def test_imgPaths(self):
        for e in imgPaths(recursive=True):
            if r"E:\OneDrive\Code\test2.png" == e: print("[+] (None, Default, True) Same folder test passed")