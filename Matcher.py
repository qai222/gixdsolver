import numpy as np
import warnings


class Matcher:
    def __init__(self, parsed_sgs_matrix, expt_peaks, cell_guess, hkl_parallel=(0, 0, 1), sg=1, hklmax=(5, 5, 5),
                 qlim=2.0, fuzzy=0.05):
        """
        :param parsed_sgs_matrix:
        :param expt_peaks:
        :param cell_guess: [a, b, c, alpha, beta, gamma] in \AA or deg
        :param hkl_parallel: index of the plane parallel to substrate
        :param sg:
        :param hklmax:
        :param qlim: limit based on detector area
        :param fuzzy:
        """
        self.fuzzy = fuzzy
        self.hkl_parallel = hkl_parallel
        self.qlim = qlim
        self.hklmax = hklmax
        self.expt_peaks = expt_peaks
        self.parsed_sgs_matrix = parsed_sgs_matrix
        self.sg = sg
        self.hkl_allowed = self.space_group_symmetry()

        self._cell_guess = cell_guess

    @property
    def cell_guess(self):
        return self._cell_guess

    @cell_guess.setter
    def cell_guess(self, paramset):
        self._cell_guess = paramset

    @property
    def how_match(self):
        found_peaks = self.expt_peaks
        theroy_peaks = self.theoryptfinder()
        fuzzy = self.fuzzy

        tot_fps = len(found_peaks)
        tot_tps = len(theroy_peaks)

        if tot_tps == 0:
            warnings.warn('theory peaks # is 0')
            return 0.0

        covered_peaks = 0
        for fp in found_peaks:
            iscover = False
            reach_alltps = False
            idx_tp = 0
            while (not iscover) and (not reach_alltps):
                tp = theroy_peaks[idx_tp]
                distance = np.linalg.norm(tp - fp)
                if idx_tp == tot_tps - 1:
                    reach_alltps = True
                if distance > fuzzy:
                    idx_tp += 1
                else:
                    iscover = True
            if iscover:
                covered_peaks += 1
        match_ratio = covered_peaks / tot_fps
        return round(match_ratio, 4)

    def theoryptfinder(self):
        """

        :return: np array of theory pts coords with val less than qlim
        """
        unit_cell = self._cell_guess
        hkl = self.hkl_allowed
        hkl_pp = self.hkl_parallel
        qlim = self.qlim

        a, b, c, alpha, beta, gamma = unit_cell
        ra = np.deg2rad(alpha)
        rb = np.deg2rad(beta)
        rg = np.deg2rad(gamma)
        vol = a * b * c * np.sqrt(
            1 + 2 * np.cos(ra) * np.cos(rb) * np.cos(rg) - np.cos(ra) ** 2 - np.cos(rb) ** 2 - np.cos(rg) ** 2)
        a_star = 2 * np.pi * b * c * np.sin(ra) / vol
        b_star = 2 * np.pi * a * c * np.sin(rb) / vol
        c_star = 2 * np.pi * a * b * np.sin(rg) / vol
        beta_star = np.arccos((np.cos(ra) * np.cos(rg) - np.cos(rb)) / abs(np.sin(ra) * np.sin(rg)))
        gamma_star = np.arccos((np.cos(ra) * np.cos(rb) - np.cos(rg)) / abs(np.sin(ra) * np.sin(rb)))

        a_matrix = np.array([[a_star], [0.0], [0.0]])
        b_matrix = np.array([[b_star * np.cos(gamma_star)], [b_star * np.sin(gamma_star)], [0.0]])
        c_matrix = np.array([[c_star * np.cos(beta_star)], [-c_star * np.sin(beta_star) * np.cos(ra)], [2 * np.pi / c]])

        g = hkl_pp[0] * a_matrix + hkl_pp[1] * b_matrix + hkl_pp[2] * c_matrix
        theory_points = []
        if np.any(g):  # the condition was ((~isreal(G) == 0) && (any(G) == 1)) #???
            phi = np.arctan2(g[1][0], g[0][0])
            chi = np.arccos(g[2][0] / np.sqrt(g[0][0] ** 2 + g[1][0] ** 2 + g[2][0] ** 2))

            r1 = [[np.cos(chi), 0.0, np.sin(-chi)], [0.0, 1.0, 0.0], [np.sin(chi), 0.0, np.cos(chi)]]
            r2 = [[np.cos(phi), np.sin(phi), 0.0], [-1 * np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]]
            r = np.matmul(r1, r2)

            ar_matrix = np.matmul(r, a_matrix)
            br_matrix = np.matmul(r, b_matrix)
            cr_matrix = np.matmul(r, c_matrix)

            for j in range(len(hkl)):
                q = hkl[j][0] * ar_matrix + hkl[j][1] * br_matrix + hkl[j][2] * cr_matrix
                qxy = np.sqrt(q[0][0] ** 2 + q[1][0] ** 2)
                qz = q[2][0]
                if 0 < qxy < qlim and 0 < qz < qlim:
                    theory_points.append([qxy, qz])
        return np.array(theory_points)

    def space_group_symmetry(self):
        """
        Anna:   assumes no special Wyckoff positions
                unique axes: monoclinic --> b; tetragonal & hexagonal --> c
                includes space groups 1-230
        :return: allowed index
        """
        hmax, kmax, lmax = self.hklmax
        spacegroup = self.sg
        sgs_matrix = self.parsed_sgs_matrix

        sg_index = int(spacegroup - 1)
        hrange = np.arange(-hmax, hmax + 1, dtype=np.float64)
        krange = np.arange(-kmax, kmax + 1, dtype=np.float64)
        lrange = np.arange(-lmax, lmax + 1, dtype=np.float64)

        hkl_start = np.array(np.meshgrid(hrange, krange, lrange)).T.reshape(-1, 3)  # Anna used RNG for this
        hkl_allow = hkl_start

        if sgs_matrix[0][sg_index] == 2.5:
            hkil = np.zeros((len(hkl_start), 4))
            hkil[:, [0, 1, 3]] = hkl_allow[:, [0, 1, 2]]
            hkil[:, 2] = -1 * (hkl_allow[:, 0] + hkl_allow[:, 1])

        # Anna:  PERMUTATIONS OF EQUIVALENT PLANES
        #       key: not permutable (1), hk permutable (2), hki cyclically permutable (2.5),
        #       ...hkl cyclically permutable (3), hkl permutable (6)

        for perm in range(int(np.ceil(sgs_matrix[0][sg_index]))):
            if sgs_matrix[0][sg_index] == 2 and perm == 1:  # hk->kh
                hkl_allow[:, [0, 1]] = hkl_allow[:, [1, 0]]

            if sgs_matrix[0][sg_index] == 2.5 and perm >= 1:  # hk(i)->ih(k) || ih(k)->ki(h)
                try:
                    hkil[:, [0, 1, 2, 3]] = hkil[:, [2, 0, 1, 3]]
                except IndexError:
                    print('error in hkil rules')
                hkl_allow[:, [0, 1, 2]] = hkil[:, [0, 1, 3]]

            if sgs_matrix[0][sg_index] >= 3 and (perm == 1 or perm == 2):  # hkl->lhk || lhk->klh
                hkl_allow[:, [0, 1, 2]] = hkl_allow[:, [2, 0, 1]]

            if sgs_matrix[0][sg_index] == 6:
                if perm == 3:
                    hkl_allow[:, [0, 1, 2]] = hkl_allow[:, [0, 2, 1]]
                elif perm >= 4:
                    hkl_allow[:, [0, 1, 2]] = hkl_allow[:, [2, 0, 1]]

            for idx in range(0, len(hkl_allow)):
                for row_index in range(1, 35):
                    if sgs_matrix[row_index][sg_index] == 1:
                        if self.reflection_rules(idx, hkl_allow, row_index):
                            hkl_allow[idx, :] = np.NaN

            if sgs_matrix[0][sg_index] == 2.5:
                hkil[:, [0, 1, 3]] = hkl_allow[:, [0, 1, 2]]
                hkil[:, 2] = -1 * (hkl_allow[:, 0] + hkl_allow[:, 1])

        # Anna: PERMUTE BACK TO ORIGINAL ORDER
        perm = int(np.ceil(sgs_matrix[0][sg_index])) - 1
        if sgs_matrix[0][sg_index] == 2 and perm == 1:  # kh->hk
            hkl_allow[:, [0, 1]] = hkl_allow[:, [1, 0]]

        if sgs_matrix[0][sg_index] == 2.5 and perm == 2:  # ki(h) ---> hk(i)
            hkil[:, [0, 1, 2, 3]] = hkil[:, [2, 0, 1, 3]]
            hkl_allow[:, [0, 1, 2]] = hkil[:, [0, 1, 3]]

        if sgs_matrix[0][sg_index] == 3 and perm == 2:  # ki(h) ---> hk(i)
            hkl_allow[:, [0, 1, 2]] = hkl_allow[:, [2, 0, 1]]

        if sgs_matrix[0][sg_index] == 6 and perm == 5:  # ki(h) ---> hk(i)
            hkl_allow[:, [0, 1, 2]] = hkl_allow[:, [0, 2, 1]]

        # Anna: remove (000)
        for i in range(len(hkl_allow)):
            if hkl_allow[i][0] == 0 and hkl_allow[i][1] == 0 and hkl_allow[i][2] == 0:
                hkl_allow[i][0] = np.NaN
                hkl_allow[i][1] = np.NaN
                hkl_allow[i][2] = np.NaN

        if sgs_matrix[0][sg_index] == 2.5:
            for i in range(len(hkil)):
                if hkil[i][0] == 0 and hkil[i][1] == 0 and hkil[i][2] == 0 and hkil[i][3] == 0:
                    hkil[i][0] = np.NaN
                    hkil[i][1] = np.NaN
                    hkil[i][2] = np.NaN
        return hkl_allow

    @staticmethod
    def reflection_rules(idx, hkl_allow, row_index):
        """
        a handy function for pulling out reflection conditions
        only used in space_group_symmetry(h_max,k_max,l_max,sg,sgs_matrix):

        :param idx: index for entry in hkl_allow, starts from 0
        :param hkl_allow: array of hkl needs to be screened
        :param row_index: row index for sgs_matrix
        :return: boolean
        """
        h = hkl_allow[idx][0]
        k = hkl_allow[idx][1]
        l = hkl_allow[idx][2]
        hk = h + k
        kl = k + l
        hl = h + l
        sgi = row_index
        if sgi == 1:
            return hk % 2 != 0 or kl % 2 != 0 or hl % 2 != 0
        elif sgi == 2:
            return (h + k + l) % 2 != 0
        elif sgi == 3:
            return hk % 2 != 0
        elif sgi == 4:
            return kl % 2 != 0
        elif sgi == 5:
            return l == 0 and (h % 2 != 0 or k % 2 != 0)
        elif sgi == 6:
            return l == 0 and hk % 2 != 0
        elif sgi == 7:
            return l == 0 and (hk % 4 != 0 or h % 2 != 0 or k % 2 != 0)
        elif sgi == 8:
            return l == 0 and h % 2 != 0
        elif sgi == 9:
            return l == 0 and k % 2 != 0
        elif sgi == 10:
            return k == 0 and (h % 2 != 0 or l % 2 != 0)
        elif sgi == 11:
            return k == 0 and hl % 2 != 0
        elif sgi == 12:
            return k == 0 and (hl % 4 != 0 or h % 2 != 0 or l % 2 != 0)
        elif sgi == 13:
            return k == 0 and h % 2 != 0
        elif sgi == 14:
            return k == 0 and l % 2 != 0
        elif sgi == 15:
            return h == 0 and (k % 2 != 0 or l % 2 != 0)
        elif sgi == 16:
            return h == 0 and kl % 2 != 0
        elif sgi == 17:
            return h == 0 and (kl % 4 != 0 or l % 2 != 0 or k % 2 != 0)
        elif sgi == 18:
            return h == 0 and k % 2 != 0
        elif sgi == 19:
            return h == 0 and l % 2 != 0
        elif sgi == 20:
            return h == k and (2 * h + l) % 4 != 0
        elif sgi == 21:
            return h == k and (h % 2 != 0 or l % 2 != 0)
        elif sgi == 22:
            return h == k and hl % 2 != 0
        elif sgi == 23:
            return h == k and l % 2 != 0
        elif sgi == 24:
            return h == -k and l % 2 != 0
        elif sgi == 25:
            return h == -k and l == 0 and h % 2 != 0
        elif sgi == 26:
            return k == 0 and l == 0 and h % 2 != 0
        elif sgi == 27:
            return k == 0 and l == 0 and h % 4 != 0
        elif sgi == 28:
            return k == 0 and h == 0 and l % 2 != 0
        elif sgi == 29:
            return k == 0 and h == 0 and l % 3 != 0
        elif sgi == 30:
            return k == 0 and h == 0 and l % 4 != 0
        elif sgi == 31:
            return k == 0 and h == 0 and l % 6 != 0
        elif sgi == 32:
            return l == 0 and h == 0 and k % 2 != 0
        elif sgi == 33:
            return l == 0 and h == 0 and k % 4 != 0
        elif sgi == 34:
            return h == k and h == l and h % 2 != 0
