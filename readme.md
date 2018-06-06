# GIWAX Indexing
Based solely on peak posistions:
- find good-fit sets of lattice parameters
- index the diffraction peaks 

## Synopsis
The raw data should be a matlab **.fig* file which can be parsed directly with [scipy.io](
https://docs.scipy.org/doc/scipy/reference/io.html). 
After the experimental peak positions are identified, 
a simulated annealing process is used to find the good-fit solutions. 
Notice the solution space may be further reduced based on unit cell transformation 
(thanks to Dr. Sean Parkin).

The theoretical positions calculation is a python translation of [DPC Toolkit](https://doi.org/10.1107/S1600576714022006)
 developed by Anna Hailey from [Lynn Loo's group](https://www.princeton.edu/cbe/people/faculty/loo/group/).
 
The simulated annealing part is an implementation of Matthew Perry's [simanneal](
https://github.com/perrygeo/simanneal) project.

## Required Packages
`scipy`

`numpy`

`cv2`

`scikit-image`

## Usage
The peak finding parameters are set in `__init__` of [Mathcer.py](./Matcher.py).
Always make a peak finding plot by:
```
python PeakPlt.py
```
to make sure these parameters are resonable.

The updating parameters are set as global vars in [GIXDSimAnneal.py](./GIXDSimAnneal.py).
To find a plasusible set of annealing parameters, uncomment the folloing 
section in [gindex_anneal.py](./gindex_anneal.py).
```
# # auto find anneal param
# print (tsp.auto(minutes=50))
```
After setting up all annealing parameters, one can do the annealing by:
```
python gindex_anneal.py
```
and the solution will be written to `anneal.out`. 
Use [dlx3anneal.js](./dlx3anneal.js) as job submission file if you are on dlx.

Once you have some solutions written in `anneal.out`, 
run 
```
python solred.py anneal.out
``` 
to find the unique solutons.



## Known Issues
1. When the crystallography plane // to substrate is set to *(00L)*, 
the diffraction pattern does not change even after a cell transformation
involving changes in *c* axis. Same thing happened when using Anna's code. 
Not sure this is physical
2. The algo for reducing solution space can be further optimized.