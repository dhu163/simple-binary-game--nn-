import numpy as np

class binary_game:
    def __init__(self, n,scores =None, model = None, alpha=1):
        self.n = n
        self.scores = scores if scores is not None else np.random.randint(10, size = 2**(n-1))
        assert len(self.scores) == 2**(n-1)
        self.binary = np.array([2**(n-2-i) for i in range(n-1)])

        self.model = model

        self.alpha = alpha

        print("Scores:", self.scores)

    def play(self, format=False):
        #start at root
        # if format is True, output is in training data format with y in the last column
        state = np.zeros(self.n, dtype = int)
        state[0]=1

        record = [state]

        while 1:
            children = self.children(state)
            if children is None:
                if format:
                    x = np.stack(record, axis=0)
                    y = np.zeros(len(record)) + self.score(state)
                    return np.c_[x, y]
                return record, self.score(state)

            state = self.choose_move(children)
            record.append(state)

    def choose_move(self, children):
        if self.model is None:
            r = np.random.randint(2)

            return children[r]

        scores = self.model(children).numpy()
        p = self.softmax(scores)
        r = np.random.rand()

        return children[int(r>p)]

    def softmax(self, scores):
        scores = np.exp(np.array(scores)*self.alpha)
        return scores[0]/(scores[0]+scores[1])

    @staticmethod
    def children(parent):
        if parent[-1]==1:
            return None
        
        level = np.max(np.nonzero(parent))
        child1 = parent.copy()
        child1[level]=0
        child1[level+1]=1
        child2 = parent.copy()
        child2[level]=1
        child2[level+1]=1
        return np.array([child1, child2])

    def dfs(self, root=None):
        if root is None:
            root = np.zeros(self.n, dtype = int)
            root[0]=1

        while 1:
            children = self.children(root)
            if children is None:
                return [root], [self.score(root)]
            dfs_left, score_left = self.dfs(children[0])
            dfs_right, score_right = self.dfs(children[1])
            p = self.softmax([score_left[0],score_right[0]])
            return [root]+dfs_left+dfs_right, [p*score_left[0] + (1-p)*score_right[0]]+score_left+score_right

    def score(self, state):
        assert state[-1]==1
        n = len(state)
        num = sum(self.binary*state[:-1])
        return self.scores[num]

b = binary_game(2)
print(b.play())