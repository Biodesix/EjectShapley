class CrastNode:
    def __init__(self) -> None:
        self.child_left = -999
        self.child_right = -999
        self.feature = -999
        self.threshold = -999
        self.value = -999
        self.sample_weight = -999
        self.depth = -999
        self.is_categorical = False
        self.left_classes = []

        # training properties
        self.score = -999
        self.entropy = -999
        self.sample_idxs = []
        self.id = -999
    
    def predict(self, X):
        if self.feature < 0:
            return -999
        if self.is_categorical:
            if X[self.feature] in self.left_classes:
                return self.child_left
            return self.child_right
        else:
            if X[self.feature] < self.threshold:
                return self.child_left
            return self.child_right

    def print(self) -> None:
        print('left: %i'%(self.child_left))
        print('right: %i'%(self.child_right))
        print('feature: %i'%(self.feature))
        print('threshold: %f'%(self.threshold))
        print('value: %s'%(self.value))
        print('depth: %i'%(self.depth))
