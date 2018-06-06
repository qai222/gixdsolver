import itertools
import math
import sys
import numpy as np


class AnnealSolution:

    def __init__(self, a, b, c, alpha, beta, gamma, howmatch):

        self.r_entry = (a, b, c, alpha, beta, gamma, howmatch)  # raw entry
        self.n_entry = self.normalize_entry(self.r_entry)      # normalize to Anna's convention
        self.mat, self.v = self.latparam2vs(self.n_entry[:-1])

    def __hash__(self):
        return 0

    def __eq__(self, other):
        if AnnealSolution.compare(self, other) and isinstance(other, AnnealSolution):
            return 1
        else:
            return 0

    @staticmethod
    def compare(ref, com, fuzzy=0.15):
        ref_m = ref.mat
        com_m = com.mat
        tof = AnnealSolution.areequi(ref_m, com_m, fuzzy)
        return tof

    @staticmethod
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
        return tuple([d[0], d[1], d[2], na, nb, ng, d[6]])

    @staticmethod
    def areequi(amat, b_target, fuzzy):
        b_lat_target = AnnealSolution.vs2latparam(b_target)
        product = itertools.product([-1, 0, 1], repeat=9)
        for it in product:
            x = np.reshape(it, (3, 3))
            b = np.matmul(amat.T, x.T).T
            b_lat = AnnealSolution.vs2latparam(b)
            if np.allclose(b_lat, b_lat_target, rtol=fuzzy):
                return True
        return False

    @staticmethod
    def unify(v):
        if np.linalg.norm(v) == 0.0:
            return [0.0, 0.0, 0.0]
        else:
            return v/np.linalg.norm(v)

    @staticmethod
    def angle_btw(v1, v2):
        v1_u = AnnealSolution.unify(v1)
        v2_u = AnnealSolution.unify(v2)
        angle_radian = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return np.rad2deg(angle_radian)

    @staticmethod
    def vs2latparam(mat):
        v1, v2, v3 = mat
        a = np.linalg.norm(v1)
        b = np.linalg.norm(v2)
        c = np.linalg.norm(v3)
        alpha = AnnealSolution.angle_btw(v2, v3)
        beta = AnnealSolution.angle_btw(v1, v3)
        gamma = AnnealSolution.angle_btw(v1, v2)
        if alpha > 90:
            alpha = 180 - alpha
        if beta > 90:
            beta = 180 - beta
        if gamma > 90:
            gamma = 180 - gamma
        latparam = np.zeros(6)
        latparam[0] = a
        latparam[1] = b
        latparam[2] = c
        latparam[3] = alpha
        latparam[4] = beta
        latparam[5] = gamma
        return latparam

    @staticmethod
    def latparam2vs(uc):
        a, b, c, alpha, beta, gamma = uc
        cosdelta = (math.cos(math.radians(alpha)) - math.cos(math.radians(beta))*math.cos(math.radians(gamma))) / \
                   (math.sin(math.radians(beta))*math.sin(math.radians(gamma)))
        sindelta = math.sqrt(1 - cosdelta**2)
        vsmat = np.zeros((3, 3))
        vsmat[0][0] = a
        vsmat[0][1] = 0.0
        vsmat[0][2] = 0.0
        vsmat[1][0] = b * math.cos(math.radians(gamma))
        vsmat[1][1] = b * math.sin(math.radians(gamma))
        vsmat[1][2] = 0.0
        vsmat[2][0] = c * math.cos(math.radians(beta))
        vsmat[2][1] = c * math.sin(math.radians(beta)) * cosdelta
        vsmat[2][2] = c * math.sin(math.radians(beta)) * sindelta
        volume = np.dot(vsmat[0], np.cross(vsmat[1], vsmat[2]))
        return vsmat, volume
