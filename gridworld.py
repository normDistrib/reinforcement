import numpy as np
import matplotlib.pyplot as plt
from utils import gridworld_policy_eval

M = 30
N = 20

def _A(s):
    d = [-1,1]
    dic1 = {
            1: "\u2193",
            -1: "\u2191"
            }
    dic2 = {
            1: "\u2192",
            -1: "\u2190"
            }
    return [[(a,s[1]),dic1[x]] for a,x in [(x+s[0],x) for x in d] if a >=0 and a<M]\
            +[[(s[0],a),dic2[x]] for a,x in [(x+s[1],x) for x in d] if a >=0 and a<N]

T = [(0,0),(5,7),(25,10),(M-1,N-1)]
gw = gridworld_policy_eval(M,N,k=15,T=T,gamma=0.9)


plt.imshow(gw)

for i in range(M):
    for j in range(N):
        if (i,j) not in T:
            dirs = _A([i,j])
            nbs = [x[0] for x in dirs]
            vals = np.array([gw[pos] for pos in nbs])
            max_vals = np.where(vals == vals[np.argmax(vals)])[0]

            for mv in max_vals:
                plt.annotate( xy=(j, i), text = dirs[mv][1])

plt.show()


