import numpy

class Tree(object):
    def __init__(self, data = None, probability = None, loc_tuple = None):
        self.data = data.cuda()    #image
        self.children = []
#        if probability != None:
#            probability = probability.amax()    #probablity is of size = (1,10) so is chaged to scalar
        self.prob = probability 
        self.mask_loc = loc_tuple

    def add_child(self, node):
        if self.children:
            index = 0
            for child in self.children:
                if node.prob <= child.prob:
                    self.children.insert(index, node)
                    return
                index = index + 1
                    
        self.children.append(node)  ##if it's the first child to add or it's the highest prob child so far
        return

    def get_depth(self):    #redundanttt maybe!!!!
        if self.parent:
            return parent.get_depth() + 1
        else:
            return 0    ####single node has depth of 0

