import utilsSTE
import numpy as np
from typing import List
from time import time
import itertools

def _score_next(q: List[int], tau, X):
    """
    copy/pasted from STE/myApp.py's getQuery
    (slightly modified)
    """
    n = X.shape[0]
    #  a, b, c = q
    # q = random_query(n) == [head, winner, loser]
    # utilsSTE.getRandomQuery == [winner, loser, head]
    # this is from utilsSTE.py#getRandomQuery and myAlg.py#get_query (which calls utilsSTE#getRandomQuery)
    # from myAlg.py#getQuery: b, c, a = q
    #                   (winner, loser, head)
    #  a, b, c = q
    b, c, a = q
    p = 0
    for i in range(n):
        p += utilsSTE.getSTETripletProbability(X[b], X[c], X[i]) * tau[a, i]

    taub = list(tau[a])
    for i in range(n):
        taub[i] = taub[i] * utilsSTE.getSTETripletProbability(X[b], X[c], X[i])
    taub = taub / sum(taub)

    tauc = list(tau[a])
    for i in range(n):
        tauc[i] = tauc[i] * utilsSTE.getSTETripletProbability(X[c], X[b], X[i])
    tauc = tauc / sum(tauc)

    entropy = -p * utilsSTE.getEntropy(taub) - (1 - p) * utilsSTE.getEntropy(tauc)
    return entropy

def run_search():
    #  n = 90
    n = 85
    d = 2
    tau = np.random.uniform(size=(n, n))
    X = np.random.uniform(size=(n, d))

    start = time()
    best = -np.inf
    deadline = time() + 50e-3
    for k in itertools.count():
        score = _score_next([0, 1, 3], tau, X)
        total = time() - start
        if score > best:
            best = score
        if time() > deadline:
            break
    print(k + 1)


if __name__ == "__main__":
    # Searches between 15 and 17 queries on my machine
    # Suppose: 1 query/sec + 2 users with a response time of 2 sec.
    # Searches have about 20 queries/search
    # So that's 20 queries/sec (if ideal scaling)
    # n_search in [30, 100, 300, 1000, 3000]
    # (if model updates take 5 secs, a total of 100 queries searched)

    for k in range(5):
        run_search()
