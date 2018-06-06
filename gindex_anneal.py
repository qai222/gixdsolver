import DiffParser
import Matcher
import GIXDSimAnneal
'''
main for indexing
'''
figname = 'sva.fig'
Cell_Init_Guess = [10., 10., 10., 90., 90., 90.]

DP = DiffParser.DiffParser(fig_name=figname)
DP.plt_foundpeaks()   # uncomment to see peaks identified
Parsed_sgs_Matrix = DP.sgs_matrix
Smooth_Erode, Expt_Peaks = DP.detect_peaks()
M = Matcher.Matcher(Parsed_sgs_Matrix, Expt_Peaks, Cell_Init_Guess, sg=1)

tsp = GIXDSimAnneal.OptProblem(Cell_Init_Guess, M)
tsp.copy_strategy = "slice"

# # auto find anneal param
# print(tsp.auto(minutes=60))
# # 'tmax': 140.0, 'tmin': 0.67, 'steps': 150000, 'updates': 100 on dlxlogin3-1

tsp.Tmax = 140
tsp.Tmin = 0.5
tsp.steps = 150000
tsp.updates = 100
state, e = tsp.anneal()
with open('anneal.out','a') as f:
   f.write('#--------------------\n')
   f.write(' '.join([str(round(i, 4)) for i in state]) + ' ' + str(round(e, 4)) + '\n')


####################
# uncomment to see timing, 0.01692512551601976 on dlxlogin3-1

# def matcher_timing():
#     import timeit
#     mycode = """M.how_match"""
#     print (timeit.timeit(stmt = mycode, number = 1000, setup="from __main__ import M") / 1000)
#
# matcher_timing()
