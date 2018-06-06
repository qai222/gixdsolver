import math
import numpy as np
import itertools


def unify(v):
    if np.linalg.norm(v) == 0.0:
        return [0.0, 0.0, 0.0]
    else:
        return v/np.linalg.norm(v)


def angle_btw(v1, v2):
    v1_u = unify(v1)
    v2_u = unify(v2)
    angle_radian = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.rad2deg(angle_radian)


def latparam2vs(uc):
    a, b, c, alpha, beta, gamma = uc
    cosdelta = (math.cos(math.radians(alpha)) - math.cos(math.radians(beta))*math.cos(math.radians(gamma))) /\
               (math.sin(math.radians(beta))*math.sin(math.radians(gamma)))
    sindelta = math.sqrt(1-cosdelta**2)
    va = a*np.array([1.0, 0.0, 0.0])
    vb = b*np.array([math.cos(math.radians(gamma)), math.sin(math.radians(gamma)), 0.0])
    vc = c*np.array([math.cos(math.radians(beta)),
                     math.sin(math.radians(beta))*cosdelta, math.sin(math.radians(beta))*sindelta])
    volume = np.dot(va, np.cross(vb, vc))
    return np.array([va, vb, vc]), volume


def vs2latparam(mat):
    v1, v2, v3 = mat
    a = l(v1)
    b = l(v2)
    c = l(v3)
    alpha = angle_btw(v2, v3)
    beta = angle_btw(v1, v3)
    gamma = angle_btw(v1, v2)
    if alpha > 90:
        alpha = 180 - alpha
    if beta > 90:
        beta = 180 - beta
    if gamma > 90:
        gamma = 180 - gamma
    return np.array([a, b, c, alpha, beta, gamma])


def l(v):
    return np.linalg.norm(v)


def intify(mat):
    intmat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mat[i][j] = int(round(mat[i][j]))
    return intmat


# # AX=B
# # A is [v1, v2, v3], X is an int mat, B is [v1, v2, v3]
def AX(A, X):
    B = np.matmul(A.T, X.T).T
    return B


def XinAXB(A, B):
    X = np.linalg.solve(A.T, B.T).T
    return X


def areequi(A, B_target, fuzzy):
    B_lat_target = vs2latparam(B_target)
    product = itertools.product([-1, 0, 1], repeat=9)
    for it in product:
        X = np.reshape(it, (3, 3))
        B = AX(A, X)
        B_lat = vs2latparam(B)
        if np.allclose(B_lat, B_lat_target, rtol=fuzzy):
            return True
    return False


def normalize_entry(d):
    if d[3] > 90.0:
        na = 180.0 - d[3]
    else:
        na = d[3]
    if d[4] > 90.0:
        nb = 180.0 - d[4]
    else:
        nb = d[4]
    if d[5] > 90.0:
        ng = 180.0 - d[5]
    else:
        ng = d[5]
    return [d[0], d[1], d[2], na, nb, ng, d[6]]
