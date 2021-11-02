class Tree:
    def __init__(self, value,part=None,rhythm = (1.0,1.0)):
        self.value = value
        self.rhythm = rhythm
        self.children = []
        self.parent = None
        self.memory = None
        self.part = part # head, body, or tail region of root

    def add_children(self, children):
        self.children += children
        for child in children:
            child.parent = self

    def get_depth(self):
        if not self.children:
            return 0
        return max([child.get_depth() for child in self.children]) + 1

    def get_dist_to_root(self):
        if not self.parent:
            return 0
        return 1+self.parent.get_dist_to_root()

    def get_subtrees_at_depth(self, depth):
        children = [self]
        while depth > 0:
            children = sum([child.children for child in children], [])
            depth = depth - 1
        return children

    def get_surface_at_depth(self, depth):
        surface = [self]
        while depth > 0:
            sub_surface = []
            for tree in surface:
                if not tree.children:
                    sub_surface.append(tree)
                else:
                    sub_surface.extend(tree.children)
            surface = sub_surface
            depth = depth - 1
        return surface

    def get_surface(self):
        max_depth = self.get_depth()
        return self.get_surface_at_depth(max_depth)

    def show(self):
        depth = self.get_depth()
        for i in range(depth+1):
            subtrees = self.get_surface_at_depth(i)
            print([x.value for x in subtrees])


class Melody(Tree):
    def __init__(self, value, latent_variables,part=None):
        super().__init__(value,part=part)
        self.surface = None
        self.latent_variables = latent_variables



tree = Tree('root')
tree1 = Tree((-5,7))
tree2 = Tree((7,5))
tree.add_children([tree1, tree2])
tree1.add_children([Tree((-5,0)), Tree((0,7))])
tree2.add_children([Tree((7,4)), Tree((4,5))])


if __name__ == '__main__':

    tree.show()
    #print(tree.get_surface()[0].parent.parent == tree)
    print('surface: ',[x.value for x in tree.get_surface()])
    print('pitch_rhythm_surface: ', [(x.value,x.rhythm) for x in tree.get_surface()])
