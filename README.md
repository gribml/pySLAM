pySLAM
======

Using PyOP2 to solve graph-based SLAM

This package requires the successful installation of the PyOP2 (www.github.com/OP2/PyOP2) package for
setting up graph-based SLAM.  A variety of other Python and non-Python packages are needed for that.
Files are read using the pandas (www.github.com/pydata/pandas) package.

Currently, the ba2D.py file solves the pose graph problem in 2D.

Some toggles:
_SAVE_DATA : save intermediate data to disk (warning, Hessian saved in dense format)

_SAVE_RESULT : save solution to disk

_DEBUG : print a variety of debugging information

_PRINT_CODE; print generated PyOP2 Kernel code

_VERBOSE : print intermediate statements of the computation

_PROFILE : run the profiler

_USE_HUBER: use the robust Huber method instead of standard sum-of-squares

_SOLVE : whether to solve the non-linear least-squares equation

SOLVER_TYPE = 'cg' (linear algebra parameter)

PRECON = 'jacobi' (linear algebra parameter)
MAX_ITER = 1 (linear algebra parameter)

