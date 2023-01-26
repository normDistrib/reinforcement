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

def gradient(n,ts,tasks,alpha=0.1,base=True):
    """
    Gradient Bandit.

    P(choosing a) = preference(a)/sum(preference(b)) for all b!=a
    Preferences are updated using stochastic aproximation of gradient ascent:
        H_t+1(a) = H_t(a) + alpha*(R_t-Baseline)*(1-P(choosing a))
        H_t+1(b) = H_t(b) - alpha*(R_t-Baseline)*(P(choosing b))

    Params:
        - n: number of actions
        - ts: number of timesteps.
        - tasks: number of n-Armed Bandit tasks generated.
        - alpha: hyper-parameter controlling the step size.
        - base: If true, using average reward as baseline. If false,
                baseline = 0.

    Returns:
        % Optimal Solutions.

    """

    optimal = np.zeros(ts)

    for _ in range(tasks):
        vals = np.random.normal(loc=4,size=n) # real values
        opt = np.argmax(vals)
        N = 1 # number of time selected an action
        baseline = 0 # baselines (always zero or average of rewards)
        preferences = np.zeros(n)
        probs = np.exp(preferences)
        probs /= np.sum(probs)

        pos = np.random.choice(n, size = 1, p = probs)[0]
        R = vals[pos] + np.random.randn(1)[0] # R_k = q(A) + random_noice
        diff = R - baseline
        if pos == opt:
            optimal[0]+=1
        preferences[pos] += alpha*diff*(1-probs[pos])
        for i in range(n):
            if i != pos:
                preferences[i] -= alpha*diff*probs[i]
        if base:
            baseline += diff/N
        N += 1

        for t in range(1,ts):

            # update probabilities
            probs = np.exp(preferences)
            probs /= np.sum(probs)

            # choose value using new distribution
            pos = np.random.choice(n, size = 1, p = probs)[0]

            # reward
            R = vals[pos] + np.random.randn(1)[0]

            diff = R - baseline

            if pos == opt:
                optimal[t]+=1

            # update preferences
            preferences[pos] += alpha*diff*(1-probs[pos])
            for i in range(n):
                if i != pos:
                    preferences[i] -= alpha*diff*probs[i]

            # update baseline
            if base:
                baseline += (R-baseline)/N
            N += 1

    return optimal/tasks


def gridworld_policy_eval(m,n,k,T,gamma=0.9):
    """
    Estimation of state value function for gridworld example.

    The policy used is a random policy which chooses any possible
    action randomly.

    Actions = {up,down,left,right}. Can´t exit the grid.
    States will be the positions in the grid.

    State-value function is upgraded k times.

    Params:
        - m,n: Grid dimensions
        - k: order of upgrade
        - T: set of terminals states
        - gamma: the discount parameter

    Returns:
        The upgraded state-value funcition for each state in form of a matrix.
    """

    #TODO: Not in-place algorithm

    V = np.zeros((m,n)) # v(s) = 0 for all s

    # Function to get possible actions for a state
    def _A(s):
        """
        Returns all s' for s.
        """
        d = [-1,1]
        return [(a,s[1]) for a in [x+s[0] for x in d] if a >=0 and a<m]\
                +[(s[0],a) for a in [x+s[1] for x in d] if a >=0 and a<n]

    def _update(s):
        next_s = _A(s)

        # pi(a|s) is same for every a in A(s)
        pi = 1/len(next_s)

        # only one s_p for a
        # p(s_p,r|s,a) = 1 because of the definition of _A(s)
        # r = -1 for all (s_p)\T
        for s_p in next_s:
            if s_p not in T:
                V[s]+=1*(-1+gamma*V[s_p])
            else:
                V[s]+=1 # 1*(1+0)

        V[s]*=pi

    for _ in range(k):
        for i in range(m):
            for j in range(n):
                s = (i,j)
                if s not in T:
                    _update(s)
    return V

if __name__ == '__main__':

    """

    # Greedy method vs e-Greedy method vs Upper Confidence Bound method
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
    """

    """

    # Gradient Bandit
    plt.plot(gradient(10,1000,500,alpha=0.1,base=True))
    plt.plot(gradient(10,1000,500,alpha=0.1,base=False))
    plt.ylim([0,1])
    plt.show()

    """

