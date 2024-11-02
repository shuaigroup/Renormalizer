import logging
from renormalizer.mps.backend import backend, np, xp

logger = logging.getLogger(__name__)

def lanczos_recurrence(Hmat, vi):
    # the current implementation uses full-diagonalization scheme
    # beta_j+1 v_j+1 = A v_j - alpha_j v_j - beta_j v_j-1
    # beta_0 = 0, v_-1 = 0, v_0 = v_i
    
    nvecs = Hmat.shape[0]
    beta = [0]
    alpha = []
    v_2 = vi
    v_1 = 0
    vset = []
    for j in range(nvecs):

        vset.append(v_2)
        v = Hmat.dot(v_2)
        alpha.append(np.vdot(v_2,v).real)
        if j == nvecs-1:
            break
        v = v - np.einsum("i,ij->j", np.array(vset).dot(v), np.array(vset)) 
        beta.append(np.linalg.norm(v))
        if np.allclose(beta[-1],0):
            print("beta approaches zero!")
        v_1 = v_2
        v_2 = v/beta[-1]

    return alpha, beta[1:], np.array(vset).T

def chain_mapping(w,gw):
    c0 = np.linalg.norm(gw)
    v1 = gw / c0
    alpha, beta, u = lanczos_recurrence(np.diag(w), v1)
    
    return alpha, beta, c0, u


