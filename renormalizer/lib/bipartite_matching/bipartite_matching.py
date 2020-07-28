# -*- coding: utf-8 -*-
# Bipartie vertex cover
# Bipartie maximum matching
# https://github.com/jilljenn/tryalgo/
# jill-jenn vie et christoph durr - 2014-2018
# adapted by jjren 

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import numpy as np

def augment(u, bigraph, visit, match):
    for v in bigraph[u]:
        if not visit[v]:
            visit[v] = True
            if match[v] is None or augment(match[v], bigraph,
                                           visit, match):
                match[v] = u       # found an augmenting path
                return True
    return False


def max_bipartite_matching(bigraph):
    """Bipartie maximum matching

    :param bigraph: adjacency list, index = vertex in U,
                                    value = neighbor list in V
    :assumption: U = V = {0, 1, 2, ..., n - 1} for n = len(bigraph)
    :returns: matching list, match[v] == u iff (u, v) in matching
    :complexity: `O(|V|*|E|)`
    """
    n = len(bigraph)               # same domain for U and V
    match = [None] * n
    for u in range(n):
        augment(u, bigraph, [False] * n, match)
    return match


def max_bipartite_matching2(bigraph):
    """Bipartie maximum matching

    :param bigraph: adjacency list, index = vertex in U,
                                    value = neighbor list in V
    :comment: U and V can have different cardinalities
    :returns: matching list, match[v] == u iff (u, v) in matching
    :complexity: `O(|V|*|E|)`
    """
    nU = len(bigraph)
    nV = max(max(adjlist, default=-1) for adjlist in bigraph) + 1
    match = [None] * nV
    for u in range(nU):
        augment(u, bigraph, [False] * nV, match)
    return match


def _alternate(u, bigraph, visitU, visitV, matchV):
    """extend alternating tree from free vertex u.
      visitU, visitV marks all vertices covered by the tree.
    """
    visitU[u] = True
    for v in bigraph[u]:
        if not visitV[v]:
            visitV[v] = True
            assert matchV[v] is not None  # otherwise match is not maximum
            _alternate(matchV[v], bigraph, visitU, visitV, matchV)

def bipartite_vertex_cover(bigraph, algo="Hopcroft-Karp"):
    r"""Bipartite minimum vertex cover by Koenig's theorem

    :param bigraph: adjacency list, index = vertex in U,
                                    value = neighbor list in V
    :comment: U and V can have different cardinalities
    :returns: boolean table for U, boolean table for V
    :comment: selected vertices form a minimum vertex cover,
              i.e. every edge is adjacent to at least one selected vertex
              and number of selected vertices is minimum
    :complexity: `O(\sqrt(|V|)*|E|)`
    """
    if algo == "Hopcroft-Karp":
        coord = [(irow,icol) for irow,cols in enumerate(bigraph) for icol in cols]
        coord = np.array(coord)
        graph = csr_matrix((np.ones(coord.shape[0]),(coord[:,0],coord[:,1])))
        matchV = maximum_bipartite_matching(graph, perm_type='row')
        matchV = [None if x==-1 else x for x in matchV]
        nU, nV = graph.shape
        assert len(matchV) == nV
    elif algo ==  "Hungarian":
        matchV = max_bipartite_matching2(bigraph)
        nU, nV = len(bigraph), len(matchV)
    else:
        assert False

    matchU = [None] * nU
    
    for v in range(nV):       # -- build the mapping from U to V
        if matchV[v] is not None:
            matchU[matchV[v]] = v
    
    def old_konig():
        visitU = [False] * nU     # -- build max alternating forest
        visitV = [False] * nV
        for u in range(nU):
            if matchU[u] is None:        # -- starting with free vertices in U
                _alternate(u, bigraph, visitU, visitV, matchV)
        inverse = [not b for b in visitU]
        return (inverse, visitV)
    
    def new_konig():
        # solve the limitation of huge number of recursive calls
        visitU = [False] * nU     # -- build max alternating forest
        visitV = [False] * nV
        wait_u = set(range(nU)) - set(matchV) 
        while len(wait_u) > 0:
            u = wait_u.pop()
            visitU[u] = True
            for v in bigraph[u]:
                if not visitV[v]:
                    visitV[v] = True
                    assert matchV[v] is not None  # otherwise match is not maximum
                    assert matchV[v] not in wait_u
                    wait_u.add(matchV[v])
        inverse = [not b for b in visitU]
        return (inverse, visitV)
    
    #res_old = old_konig()
    res_new = new_konig()
    #assert res_old == res_new
    return res_new
