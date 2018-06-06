from simanneal import Annealer
import random


# anneal parameters
abc_moving_lim = 0.2
abg_moving_lim = 0.5
alim = [3.0, 20.0]
blim = [3.0, 20.0]
clim = [3.0, 20.0]
alphalim = [60.0, 120.0]
betalim = [60.0, 120.0]
gammalim = [60.0, 120.0]


class OptProblem(Annealer):

    def __init__(self, state, matcher):
        super(OptProblem, self).__init__(state)
        self.matcher = matcher

    def move(self):

        delta = random.uniform(-abc_moving_lim, abc_moving_lim)
        if alim[0] < self.state[0] + delta < alim[1]:
            self.state[0] = self.state[0] + delta
        else:
            self.state[0] = self.state[0] - delta

        delta = random.uniform(-abc_moving_lim, abc_moving_lim)
        if blim[0] < self.state[1] + delta < blim[1]:
            self.state[1] = self.state[1] + delta
        else:
            self.state[1] = self.state[1] - delta

        delta = random.uniform(-abc_moving_lim, abc_moving_lim)
        if clim[0] < self.state[2] + delta < clim[1]:
            self.state[2] = self.state[2] + delta
        else:
            self.state[2] = self.state[2] - delta

        delta = random.uniform(-abg_moving_lim, abg_moving_lim)
        if alphalim[0] < self.state[3] + delta < alphalim[1]:
            self.state[3] = self.state[3] + delta
        else:
            self.state[3] = self.state[3] - delta

        delta = random.uniform(-abg_moving_lim, abg_moving_lim)
        if betalim[0] < self.state[4] + delta < betalim[1]:
            self.state[4] = self.state[4] + delta
        else:
            self.state[4] = self.state[4] - delta

        delta = random.uniform(-abg_moving_lim, abg_moving_lim)
        if gammalim[0] < self.state[5] + delta < gammalim[1]:
            self.state[5] = self.state[5] + delta
        else:
            self.state[5] = self.state[5] - delta

    def energy(self):
        self.matcher.cell_guess = self.state
        e = (1 - self.matcher.how_match) * 100.0
        return e
