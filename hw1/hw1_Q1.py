import numpy as np
np.set_printoptions(precision=2, suppress=True)

# Settings. order : FB, C1, C2, C3, Pb, Ps, Sl
transition = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 0.0],
    [0.0, 0.2, 0.4, 0.4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
])
reward = np.array(
    [-1, -2, -2, -2, 1, 10, 0]
)
num_iter = 0
gamma = 1.0
end_condition = 0.0001

# q1. state value
val = np.zeros((7,))
while True:
    new_val = reward + gamma*np.dot(transition, val)
    if sum(abs(val - new_val)) < end_condition:
        break
    else:
        val = new_val
        num_iter += 1
print("\n[q1] State Values [FB, C1, C2, C3, Pb, Ps, Sl]: \n", val, '\n', sep='')

# inverse matrix
print("State Values calculated by linear algebra:")
print(np.dot(np.linalg.inv(np.eye(7) - gamma*transition), reward))

# q2. action value
action_val = np.vstack([reward + gamma*val]*7)*(transition > 0)
print("\n[q2] Action values calculated deterministically:\n", action_val, '\n', sep='')

# q3. number of iterations
print(f"[q3] number of iterations: {num_iter}\n")