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
        Rewards averaged between total tasks, % Optimal Solutions.

    """

    total_rewards = np.zeros(ts)
    optimal = np.zeros(ts)

    for _ in range(tasks):
        vals = np.random.randn(n) # real values
        opt = np.argmax(vals)
        N = np.ones(n) # number of time selected an action
        estimates = np.zeros(n) # estimations of values for each action

        pos = np.random.randint(0,n)
        if pos == opt:
            optimal[0]+=1

        R = vals[pos] + np.random.randn(1)[0] # R_k = q(A) + random_noice
        estimates[pos] = estimates[pos]+1/N[pos]*(R-estimates[pos])
        total_rewards[0]+=R
        N[pos] += 1

        for t in range(1,ts):
            pos = np.random.choice([np.random.randint(0,n),np.argmax(estimates)],
                                   size = 1,
                                   p = [e,1-e])[0]
            N[pos] += 1
            R = vals[pos] + np.random.randn(1)[0]
            if pos == opt:
                optimal[t]+=1
            estimates[pos] = estimates[pos]+1/N[pos]*(R-estimates[pos])
            total_rewards[t]+=R

    return total_rewards/tasks, optimal/tasks

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
        Rewards averaged between total tasks, % Optimal Solutions.

    """

    total_rewards = np.zeros(ts)
    optimal = np.zeros(ts)

    for _ in range(tasks):
        vals = np.random.randn(n) # real values
        opt = np.argmax(vals)
        N = np.ones(n) # number of time selected an action
        estimates = np.zeros(n) # estimations of values for each action

        pos = np.random.randint(0,n)
        R = vals[pos] + np.random.randn(1)[0] # R_k = q(A) + random_noice
        if pos == opt:
            optimal[0]+=1
        estimates[pos] = estimates[pos]+1/N[pos]*(R-estimates[pos])
        total_rewards[0]+=R
        N[pos] += 1

        for t in range(1,ts):
            pos = np.argmax(estimates + c*np.sqrt(np.log(t)/N))
            N[pos] += 1
            R = vals[pos] + np.random.randn(1)[0]
            if pos == opt:
                optimal[t]+=1
            estimates[pos] = estimates[pos]+1/N[pos]*(R-estimates[pos])
            total_rewards[t]+=R

    return total_rewards/tasks, optimal/tasks


if __name__ == '__main__':

    fig, axs = plt.subplots(2,1)
    greedy_avg, greedy_optimal = greedy(10,1000,500,e=0)
    e_greedy_avg, e_greedy_optimal = greedy(10,1000,500,e=0.1)
    ucb_avg, ucb_optimal = UCB(10,1000,500,c=2)
    axs[0].plot(greedy_avg,label='greedy. e = 0',linewidth=1) # greedy method
    axs[0].plot(e_greedy_avg,label='e-greedy. e = 0.1',linewidth=1) # e-greedy method
    axs[0].plot(ucb_avg,label='UCB. c = 2',linewidth=1) # UCB method

    axs[1].plot(greedy_optimal,label='greedy. e = 0',linewidth=1) # greedy method
    axs[1].plot(e_greedy_optimal,label='e-greedy. e = 0.1',linewidth=1) # e-greedy method
    axs[1].plot(ucb_optimal,label='UCB. c = 2',linewidth=1) # UCB method

    axs[1].set_ylim([0,1])

    plt.legend()
    plt.show()
