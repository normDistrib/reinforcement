import numpy as np
import matplotlib.pyplot as plt


def greedy(n,ts,tasks,e=0.01):
    """
    n-Armed Bandit with value estimation by sample-average of rewards.

    Params:
        - n: number of actions
        - ts: number of timesteps.
        - tasks: number of n-Armed Bandit tasks generated.
        - e: probability of choosing randomly between actions:

    Returns:
        Rewards averaged between total tasks.

    """

    total_rewards = np.zeros(ts)

    for _ in range(tasks):
        vals = np.random.randn(n) # real values
        N = np.ones(n) # number of time selected an action
        estimates = np.zeros(n) # estimations of values for each action
        rewards = [] # rewards obtained at each step

        pos = np.random.randint(0,n)
        R = vals[pos] + np.random.randn(1)[0] # R_k = q(A) + random_noice
        estimates[pos] = estimates[pos]+1/N[pos]*(R-estimates[pos])
        rewards.append(R)
        N[pos] += 1

        for _ in range(1,ts):
            pos = np.random.choice([np.random.randint(0,n),np.argmax(estimates)],
                                   size = 1,
                                   p = [e,1-e])[0]
            N[pos] += 1
            R = vals[pos] + np.random.randn(1)[0]
            estimates[pos] = estimates[pos]+1/N[pos]*(R-estimates[pos])
            rewards.append(R)

        total_rewards += np.array(rewards)

    return total_rewards/tasks

def UCB(n,ts,tasks,c):
    """
    Upper-Confidence-Band technique.

    Select action A_t = argmax(Q_t(a) + c*sqrt(ln t/ N_t(a))).
    The latter part is the uncertainty level of the action's value.
    When selected multiple times, uncertainty goes down. When not chosen,
    uncertainty goes up and it is more probable to select.

    Params:
        - n: number of actions
        - ts: number of timesteps.
        - tasks: number of n-Armed Bandit tasks generated.
        - c: hyper-parameter to control uncertainty weight.

    Returns:
        Rewards averaged between total tasks.

    """

    total_rewards = np.zeros(ts)

    for _ in range(tasks):
        vals = np.random.randn(n) # real values
        N = np.ones(n) # number of time selected an action
        estimates = np.zeros(n) # estimations of values for each action
        rewards = [] # rewards obtained at each step

        pos = np.random.randint(0,n)
        R = vals[pos] + np.random.randn(1)[0] # R_k = q(A) + random_noice
        estimates[pos] = estimates[pos]+1/N[pos]*(R-estimates[pos])
        rewards.append(R)
        N[pos] += 1

        for t in range(1,ts):
            pos = np.argmax(estimates + c*np.sqrt(np.log(t)/N))
            N[pos] += 1
            R = vals[pos] + np.random.randn(1)[0]
            estimates[pos] = estimates[pos]+1/N[pos]*(R-estimates[pos])
            rewards.append(R)

        total_rewards += np.array(rewards)

    return total_rewards/tasks


if __name__ == '__main__':
    #plt.plot(greedy(10,1000,500,0)) # greedy method
    #plt.plot(greedy(10,1000,500,e=0.01))
    plt.plot(greedy(10,1000,500,e=0.1),label='e-greedy. e = 0.1')
    plt.plot(UCB(10,1000,500,c=2),label='UCB. c = 2')
    plt.legend()
    plt.show()
