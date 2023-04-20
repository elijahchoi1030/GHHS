import numpy as np
np.set_printoptions(precision=2, suppress=True)

class env():
    def __init__(self, size, terminators, update_style, living_reward=-0.1, gamma=1, end_cond=0.0001):
        self.size = size
        self.value = np.zeros(size)
        self.terminators = terminators
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.reward = living_reward
        self.gamma = gamma
        self.end_cond = end_cond
        self.num_iter = 0
        for trm in self.terminators:
            self.value[trm['place']] = trm['val']
        if update_style == 'Value Iteration':
            self.updateStyle = self.valueIteration
        elif update_style == 'Policy Evaluation':
            self.updateStyle = self.policyEvaluation

    def valueIteration(self, next_state):
        return max(next_state)
    
    def policyEvaluation(self, next_state):
        return sum(next_state)/len(self.actions)
    
    def getNextState(self, state, action):
        nextState = [state[0]+action[0], state[1]+action[1]]
        if nextState[0] < 0:
            nextState[0] = 0
        if nextState[0] >= self.size[0]:
            nextState[0] = self.size[0]-1
        if nextState[1] < 0:
            nextState[1] = 0
        if nextState[1] >= self.size[1]:
            nextState[1] = self.size[1]-1
        return tuple(nextState)
    
    def updateValue(self):
        new_value = np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                next_state = []
                for act in self.actions:
                    next_state.append(self.reward + self.gamma * self.value[self.getNextState((i, j), act)])
                new_value[i, j] = self.updateStyle(next_state)
        for trm in self.terminators:
            new_value[trm['place']] = trm['val']
        self.num_iter += 1
        if np.sum(abs(self.value - new_value)) < self.end_cond:
            self.end_cond = None
        self.value = new_value

    def getPolicy(self):
        arrow = ('↓', '↑', '→', '←')
        policy = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                next_state = []
                for act in self.actions:
                    next_state.append(self.value[self.getNextState((i, j), act)])
                policy.append(arrow[np.argmax(next_state)])
        policy = np.array(policy).reshape(self.size)
        for trm in self.terminators:
            if trm['val'] > 0:
                policy[trm['place']] = '□'
            else:
                policy[trm['place']] = '■'
        return policy

    def printValues(self):
        print("Values", " "*24, "Policies", sep='')
        policy = self.getPolicy()
        for i in range(len(self.value)):
            print(self.value[i], " "*(30-len(np.array2string(self.value[i]))), \
                    policy[i], sep='')

    def runSimulation(self, check):
        while self.end_cond is not None:
            if self.num_iter in check:
                print(f"num iter: {self.num_iter}")
                self.printValues()
                print()
            self.updateValue()
        print(f"Learning ended in iteration {self.num_iter}")
        self.printValues()


# Policy Evaluation
# Example
print("Policy Evaluation: example")
PE_ex = env((4, 4), [{'place':(0, 0), 'val':0}, {'place':(3, 3), 'val':0}], \
               "Policy Evaluation", living_reward=-1)
PE_ex.runSimulation([0, 1, 2, 3, 10])

# Question 2
print("\n\nPolicy Evaluation: question 2")
PE_q2 = env((4, 4), [{'place':(0, 3), 'val':1}, {'place':(1, 3), 'val':-1}], \
               "Policy Evaluation")
PE_q2.runSimulation([1, 5, 10])


# Value Iteration
# Example
print("\n\nValue Iteration: example")
VI_ex = env((4, 4), [{'place':(0, 0), 'val':0}], "Value Iteration", living_reward=-1)
VI_ex.runSimulation([1, 2, 3, 4, 5, 6])

# Question 2
print("\n\nValue Iteration: question 2")
VI_q2 = env((4, 4), [{'place':(0, 3), 'val':1}, {'place':(1, 3), 'val':-1}], \
               "Value Iteration")
VI_q2.runSimulation([1, 5, 10])
