#!/usr/bin/env python

## code to solve 2D pose-only bundle adjustment using PyOP2
import numpy as np
import ba_data
import balib
from pyop2 import op2, petsc_base
import pprint as pp
import cProfile
import pstats
import sympy
from sympy.utilities.codegen import codegen
from sympy.functions.elementary.piecewise import Piecewise
import time
from functools import partial

# these get used as inputs into Kernels and should not be used otherwise
reserved_names = ['estimate', 'poses', 'err', 'omega', 'sse', 'J', 'H', 'i',
                  'j', 'rhs_vector', 'idx', 'idx1', 'idx2']

_SAVE_DATA = True
_SAVE_RESULT = True
_DEBUG = False
_PRINT_CODE = False
_VERBOSE = False
_PROFILE = False
_SOLVE = False
SOLVER_TYPE = 'cg'
PRECON = 'jacobi'
MAX_ITER = 1

_expr_count = 0
vertices, edges = ba_data.quickLoad(ba_data.MANHATTAN)
if _VERBOSE:
    print vertices.head()
    print edges.head()

POSES_PER_CONSTRAINT = 2

POSES_DIM = balib.dim_lookup[vertices.label.unique()[0]]
CONSTRAINT_DIM = balib.dim_lookup[edges.label.unique()[0]]
OMEGA_DOF = balib.omega_dof(CONSTRAINT_DIM)

NUM_POSES = len(vertices)
NUM_CONSTRAINTS = len(edges)

op2.init(backend='sequential')


def testKernel(data=None, op2set=None, op2map=None):
    """ for testing purposes, it just prints out the data Dat array
        (assuming) it has dimension 3 in its Map (op2map).  The kernel
        iterates over the Set op2set, if no arguments are passed in,
        this creates dummy values and prints them out
    """
    if not op2set:
        op2set = op2.Set(5, 'fromset')
    if not op2map:
        op2toset = op2.Set(4, 'toset')
        npmapping = np.array([0, 1, 1, 2, 2, 3, 3, 1, 3, 2], np.uint32)
        print '-' * 80
        print 'mapping: ', npmapping, npmapping.dtype, npmapping.shape
        op2map = op2.Map(op2set, op2toset, 2, npmapping, 'testmap')
    if not data:
        numpydata = np.array([[0, 1, 1],
                              [1, 2, 2],
                              [2, 3, 3],
                              [3, 4, 4]], dtype=np.float64)
        print '-' * 80
        print 'data:'
        print numpydata
        data = op2.Dat(op2toset ** 3, numpydata, np.float64, 'testdata')

    test = op2.Kernel("""
    void test_kernel(double *x[3])
    {
        std::cout << " " << x[0][0] << " " << x[0][1] << " " << x[0][2];
        std::cout << " : " << x[1][0] << " " << x[1][1] << " " << x[1][2]
                  << std::endl;
    }
    """, 'test_kernel')

    print '-' * 80
    print 'PyOP2 output:'
    op2.par_loop(test, op2set, data(op2map, op2.READ))
    print '-' * 80


def preprocess():
    """ Initial step to create PyOP2 Sets and Maps with correct dimensions
        to create indirection for the graph problem:
        1) poses is a Set of all poses (parameter blocks)
        2) constraints is a Set of all measurements (edges between pose blocks)
        3) constraints_to_poses maps a constraint to 2 poses
        4) constraints_to_constraints maps a constraint to the right number
            of data based on dimension of constraints_to_poses
        5) poses_to_poses maps poses to data based on dimension of poses
    """
    poses = op2.Set(NUM_POSES, 'poses')
    constraints = op2.Set(NUM_CONSTRAINTS, 'constraints')

    if _VERBOSE:
        print '2D BA: %d poses and %d constraints between them' \
              % (NUM_POSES, NUM_CONSTRAINTS)
        print '2D BA: pose dimension %d and constraint dimensions %d' \
              % (POSES_DIM, CONSTRAINT_DIM)

    edgemappings = edges[['from_v', 'to_v']].values.reshape(POSES_PER_CONSTRAINT * NUM_CONSTRAINTS)

    if _DEBUG:
        print 'ba edges:', edgemappings, edgemappings.dtype, edgemappings.shape
    constraints_to_poses = op2.Map(constraints,
                                   poses,
                                   POSES_PER_CONSTRAINT,
                                   edgemappings,
                                   'constraints_to_poses')

    constraints_to_constraints = op2.Map(constraints,
                                         constraints,
                                         CONSTRAINT_DIM,
                                         balib.identity_map(NUM_CONSTRAINTS, CONSTRAINT_DIM),
                                         'constraints_to_constraints')

    poses_to_poses = op2.Map(poses,
                             poses,
                             POSES_DIM,
                             balib.identity_map(NUM_POSES, POSES_DIM),
                             'poses_to_poses')

    # this is needed to update hessian initial condition values
    initial = op2.Set(1, 'initial')
    initial_to_poses = op2.Map(initial, poses, 1, [0], 'initial_to_poses')
    # initial_to_constraints = op2.Set(  initial)

    return (poses, constraints, initial, constraints_to_poses,
            constraints_to_constraints, poses_to_poses, initial_to_poses)


def setupData(poses, constraints):
    """ setting up Dat objects from the loaded vertex and edge DataFrames
        1) err is the large vector of measurement errors
        2) estimate is the estimate we will calculate based on parameters
        3) measurement is the actual measurement between vertices
        4) jacobian blocks are used to build rhs_vec and hessian
        5) omega_blocks is a static quantity from the edges DataFrame
        6) rhs_vec will hold the vector of values for the NLS equation
        7) sse is sum of squares we use to evaluate convergence
    """
    err = op2.Dat(dataset=constraints ** CONSTRAINT_DIM,
                  data=None,
                  dtype=np.float64,
                  name='e')
    estimate = op2.Dat(dataset=constraints ** CONSTRAINT_DIM,
                       data=None,
                       dtype=np.float64,
                       name='estimate')

    pose_slice = ['dim%d' % i for i in xrange(1, POSES_DIM + 1)]
    x = op2.Dat(dataset=poses ** POSES_DIM,
                data=vertices[pose_slice].values.reshape(POSES_DIM, NUM_POSES),
                dtype=np.float64,
                name='x')
    
    if _DEBUG:
        print '-' * 80
        print 'ba vertices:'
        vertex_data = vertices[pose_slice].values
        print vertex_data
        print vertex_data.shape

    meas_slice = ['meas%d' % i for i in xrange(1, CONSTRAINT_DIM + 1)]
    measurement = op2.Dat(dataset=constraints ** CONSTRAINT_DIM,
                          data=edges[meas_slice].values.reshape(CONSTRAINT_DIM, NUM_CONSTRAINTS),
                          dtype=np.float64,
                          name='measurement')

    jacobian_blocks = op2.Dat(dataset=constraints ** (CONSTRAINT_DIM * POSES_DIM * POSES_PER_CONSTRAINT),
                              data=None,
                              dtype=np.float64,
                              name='jacobian_blocks')

    omega_slice = ['meas%d' % i for i in xrange(CONSTRAINT_DIM + 1,
                                                CONSTRAINT_DIM + OMEGA_DOF + 1)]
    omega_blocks = op2.Dat(dataset=constraints ** OMEGA_DOF,
                           data=edges[omega_slice].values.reshape(OMEGA_DOF, NUM_CONSTRAINTS),
                           dtype=np.float64,
                           name='omega_blocks')

    rhs_vec = op2.Dat(poses ** POSES_DIM, dtype=np.float64, name='rhs_vec')
    dx = op2.Dat(dataset=poses ** POSES_DIM, dtype=np.float64, name='dx')
    # scalar value which is the sum-of-squares calcuation
    sse = op2.Global(dim=1, data=[0.0], dtype=np.float64)
    sse_arr = op2.Dat(dataset=constraints ** 1, dtype=np.float64, name='array')

    return (err, x, estimate, measurement, jacobian_blocks, omega_blocks,
            rhs_vec, dx, sse, sse_arr)


def setuphessian(poses, constraints_to_poses):
    """ define the sparsity pattern of the hessian and return the Mat """
    hess_sparsity = op2.Sparsity((poses ** POSES_DIM, poses ** POSES_DIM),
                                 (constraints_to_poses, constraints_to_poses),
                                 'hess_sparsity')

    if _VERBOSE:
        print 'hessian has dimension %d by %d' % (NUM_POSES * POSES_DIM, NUM_POSES * POSES_DIM)
    return petsc_base.Mat(hess_sparsity, np.float64, 'hess_matrix')


def generateEstimateCode(name, funcs):
    """pass in a function name and a CONSTRAINT_DIM number of functions in a
       list or a tuple to apply that function to each dimension;  the function
       should be a sympy expression we can generate C-code with;  this function
       inefficiently processes strings, but does this O(1) times.
       called as follows:
       op2.par_loop(estimate_poses, pose_constraints,
                    pose_dat(constraints_to_poses, op2.READ),
                    estimate(op2.IdentityMap, op2.WRITE))
    """

    if not hasattr(funcs, '__len__'):
        funcs = [funcs] * CONSTRAINT_DIM

    pose1_symbols = sympy.symbols(', '.join(['poses[0][%d]' % i for i in xrange(POSES_DIM)]))
    pose2_symbols = sympy.symbols(', '.join(['poses[1][%d]' % i for i in xrange(POSES_DIM)]))
    
    code = []
    for i in xrange(CONSTRAINT_DIM):
        c_code_lines = parse_return(generate_from_sympy(funcs[i](*(pose1_symbols + pose2_symbols)))).split('\n')
        for j, line in enumerate(c_code_lines):
            if (line[-1] != ';') and ('{' not in line and '}' not in line):
                c_code_lines[j] = 'estimate[%d] = %s;' % (i, line.strip())
        code.append('\n'.join(c_code_lines))

    estimate_code = """
    void %(name)s(double * poses[%(p_dim)d], double estimate[%(c_dim)d])
    {
        %(comment)s std::cout << "pose1: " << poses[0][0] << ", " <<
        %(comment)s poses[0][1] << ", " << poses[0][2] << "; pose2: " <<
        %(comment)s poses[1][0] << ", " << poses[1][1] << ", " <<
        %(comment)s poses[1][2] << std::endl;
        %(code)s
        %(comment)s std::cout << "estimate: " << estimate[0] << " " <<
        %(comment)s estimate[1] << " " << estimate[2] << std::endl;
    }
    """ % {'name': name,
           'p_dim': POSES_DIM,
           'c_dim': CONSTRAINT_DIM,
           'code': '\n'.join(code),
           'comment': '//' if _DEBUG else '//'}

    if _PRINT_CODE:
        print estimate_code
    return op2.Kernel(estimate_code, name)


def generateErrorCode(name, funcs):
    """ pass in a function name and CONSTRAINT_DIM number of functions as a
        list or tuple and get back a PyOP2 Kernel object containing the code
        to calculate error vector
        called as follows:
        op2.par_loop(pose_error, pose_constraints,
                     e(op2.IdentityMap, op2.WRITE),
                     estimate(op2.IdentityMap, op2.READ),
                     measurement(op2.IdentityMap, op2.READ))
    """

    if not hasattr(funcs, '__len__'):
        funcs = [funcs] * CONSTRAINT_DIM

    est_symbols = sympy.symbols(', '.join(['estimate[%d]' % i for i in xrange(CONSTRAINT_DIM)]))
    meas_symbols = sympy.symbols(', '.join(['measurement[%d]' % i for i in xrange(CONSTRAINT_DIM)]))
    code = []
    for i in xrange(CONSTRAINT_DIM):
        c_code_lines = parse_return(generate_from_sympy(funcs[i](*(est_symbols + meas_symbols)))).split('\n')
        for j, line in enumerate(c_code_lines):
            if (line[-1] != ';') and ('{' not in line and '}' not in line):
                c_code_lines[j] = 'err[%d] = %s;' % (i, line.strip())
        code.append('\n'.join(c_code_lines))

    error_code = """
    void %(name)s(double estimate[%(c_dim)s],
                  double measurement[%(c_dim)s],
                  double err[%(c_dim)s])
    {
        %(comment)s std::cout << "estimate " << estimate[0] << " " <<
        %(comment)s estimate[1] << " " << estimate[2] << std::endl;
        %(comment)s std::cout << "measurement " << measurement[0] << " " <<
        %(comment)s measurement[1] << " " << measurement[2] << std::endl;
        %(code)s
        %(comment)s std::cout << "error " << err[0] << " " << err[1] <<
        %(comment)s " " << err[2] << std::endl;
    }
    """ % {'name': name,
           'c_dim': CONSTRAINT_DIM,
           'code': '\n'.join(code),
           'comment': '' if _DEBUG else '//'}

    if _PRINT_CODE:
        print error_code
    return op2.Kernel(error_code, name)


def generateSSECode(name):
    """pass in a name and get back a Kernel to compute the sum of squares
       called in the following way:
       op2.par_loop(kernel, constraints,
                    e(op2.IdentityMap, op2.READ),
                    omega_blocks(op2.IdentityMap, op2.READ),
                    F(op2.INC))
    """

    ij = [(i, j) for i in xrange(CONSTRAINT_DIM)
          for j in xrange(CONSTRAINT_DIM) if j >= i]
    unrolled_loop = []
    for idx, (i, j) in enumerate(ij):
        unrolled_loop.append('%f * err[%d]*omega[%d]*err[%d]' %
                             (2 - (i == j), i, idx, j))

    sse_code = """
    void %(name)s(double err[%(c_dim)d], double omega[%(omega_dim)d],
                  double array[1], double *sse) //, double array[1])
    {
        array[0] = %(loop)s;
        *sse += array[0];
    }
    """ % {'name': name,
           'c_dim': CONSTRAINT_DIM,
           'omega_dim': OMEGA_DOF,
           'comment': '//',
           'loop': ' + '.join(unrolled_loop)}
    
    if _PRINT_CODE:
        print sse_code
    return op2.Kernel(sse_code, name)


def generateJacobianCode(name, funcs):
    """ pass in a name string and the error function having CONSTRAINT_DIM
        dimensions and get back a kernel which computes the generic jacobian
        block determined by the Jacobian of the funcs wrt each input;  The
        Jacobian block has dimensions:
            CONSTRAINT_DIM x (POSES_PER_CONSTRAINT * POSES_DIM)
        call made as follows:
        op2.par_loop(calc_jac_blocks, pose_constraints,
                     x(constraints_to_poses, op2.READ),
                     jacobian_blocks(op2.IdentityMap, op2.RW))
    """
    
    if not hasattr(funcs, '__len__'):
        funcs = [funcs] * CONSTRAINT_DIM

    poses_symbols = []
    for k in xrange(POSES_PER_CONSTRAINT):
        poses_symbols.extend(sympy.symbols(', '.join(['poses[%d][%d]' %
                             (k, i) for i in xrange(POSES_DIM)])))

    code = []
    for i in xrange(POSES_PER_CONSTRAINT):
        for j in xrange(CONSTRAINT_DIM):
            func = funcs[j]
            for k in xrange(POSES_DIM):
                var = poses_symbols[i * CONSTRAINT_DIM + k]
                c_code_lines = parse_return(generate_from_sympy(func(*(poses_symbols)), True, var)).split('\n')
                for h, line in enumerate(c_code_lines):
                    if (line[-1] != ';') and ('{' not in line and '}' not in line):
                        c_code_lines[h] = 'J[%d] = %s;' % (i * CONSTRAINT_DIM * POSES_DIM + j * CONSTRAINT_DIM + k, line.strip())
                code.append('\n'.join(c_code_lines))

    calc_blocks_code = """
    void %(name)s(double * poses[%(p_dim)d], double J[%(j_dim)d]) {
        %(code)s
    }
    """ % {'name': name,
           'p_dim': POSES_DIM,
           'j_dim': CONSTRAINT_DIM * POSES_DIM * POSES_PER_CONSTRAINT,
           'code': '\n' + '\n'.join(code)}

    if _PRINT_CODE:
        print calc_blocks_code
    return op2.Kernel(calc_blocks_code, name)


def generateRHSCode(name):
    """ pass in a name and get back a PyOP2 kernel which computes the vector:
        J-transpose * Omega * e
        This is general enough to allow situations where we have hypergraphs,
        i.e. the number of 'vertices' is not equal 2 per 'edge'
        called in the following way:
        op2.par_loop(rhs, constraints,
                     err(op2.IdentityMap, op2.READ),
                     jacobian_blocks(op2.IdentityMap, op2.READ),
                     omega(op2.IdentityMap, op2.READ),
                     rhs_vec(constraints_to_poses, op2.INC))
    """

    # code for omega * e; do this first because it is independend of poses
    omega_ij = [(i, j) for i in xrange(CONSTRAINT_DIM)
                for j in xrange(CONSTRAINT_DIM) if j >= i]
    omega_err = []
    for i in xrange(CONSTRAINT_DIM):
        row = []
        for j in xrange(CONSTRAINT_DIM):
            idx = omega_ij.index((j, i)) if i > j else omega_ij.index((i, j))
            row.append('omega[%d] * err[%d]' % (idx, j))
        omega_err.append('omega_times_err[%d] = %s;' % (i, ' + '.join(row)))

    code = []
    for i in xrange(CONSTRAINT_DIM):
        row = []
        for j in xrange(POSES_DIM):
            row.append('J[%d*i + %d] * omega_times_err[%d]' %
                      (CONSTRAINT_DIM * POSES_DIM, j * CONSTRAINT_DIM + i, j))
        code.append('rhs_vector[i][%d] -= %s;' % (i, ' + '.join(row)))
    rhs_code = """
    void %(name)s(double err[%(c_dim)d], double J[%(j_dim)d],
                  double omega[%(o_dim)d], double * rhs_vector[%(p_dim)d])
    {
        double omega_times_err[%(c_dim)d];
        %(omega_err)s
        int i = 0;
        for ( ; i < %(poses_per_constraint)d; ++i ) {
            %(code)s
        }
    }
    """ % {'name': name,
           'poses_per_constraint': POSES_PER_CONSTRAINT,
           'j_dim': CONSTRAINT_DIM * POSES_DIM * POSES_PER_CONSTRAINT,
           'j_subblock_dim': CONSTRAINT_DIM * POSES_DIM,
           'o_dim': OMEGA_DOF,
           'p_dim': POSES_DIM,
           'c_dim': CONSTRAINT_DIM,
           'omega_err': '\n'.join(omega_err),
           'code': '\n'.join(code)}

    if _PRINT_CODE:
        print rhs_code
    return op2.Kernel(rhs_code, name)


def generatehessianCode(name, lm_param):
    """pass in a name string and a float representing the Levenberg-Marquardt
       parameter for H + lm_param * D
       The hessian is a NUM_POSES*POSES_DIM by NUM_POSES*POSES_DIM square
       matrix but is calculated by iterating over constraints, since that is
       how the jacobian blocks are obtained;
       called this way:
       op2.par_loop(hessian, constraints(POSES_DIM,POSES_DIM),
                    hessian_mat((constraints_to_poses[op2.i[0]],
                               constraints_to_poses[op2.i[1]]), op2.INC),
                    jacobian_blocks(op2.IdentityMap, op2.READ))

       also returns kernels to update the diagonal by the lm_param value and
       the initial condition (first pose)
    """
    JBLOCK_SIZE = POSES_DIM * CONSTRAINT_DIM

    posdef_partial = partial(normal_to_posdef, dim=CONSTRAINT_DIM)

    block_code = ['j_t_omega[%d] = %s;' %
                  (i * CONSTRAINT_DIM + j, dotproduct('J', i, 1, 'omega',
                                                      j, CONSTRAINT_DIM,
                                                      CONSTRAINT_DIM,
                                                      idxFunc2=posdef_partial,
                                                      prefix1='%d*j+' % (JBLOCK_SIZE)))
                  for i in xrange(POSES_DIM) for j in xrange(POSES_DIM)]

    update = ['H[%d][%d] += %s;' % (i, j, dotproduct('j_t_omega', i * CONSTRAINT_DIM, 1,
                                                     'J', j, CONSTRAINT_DIM, CONSTRAINT_DIM, prefix2='%d*i+' % (JBLOCK_SIZE)))
              for i in xrange(POSES_DIM) for j in xrange(POSES_DIM)]

    hessian_code = """
    void %(name)s(double J[%(j_dim)d], double omega[%(o_dim)d],
                  double H[%(p_dim)d][%(p_dim)d], int i, int j)
    {
        double j_t_omega[%(j_t_omega_dim)d];
        %(jacT_times_omega)s
        %(update)s
    }
    """ % {'name': name,
           'j_t_omega_dim': CONSTRAINT_DIM * POSES_DIM,
           'p_dim': POSES_DIM,
           'c_dim': CONSTRAINT_DIM,
           'o_dim': OMEGA_DOF,
           'poses_per_constraint': POSES_PER_CONSTRAINT,
           'jacT_times_omega': '\n'.join(block_code),
           'update': '\n'.join(update),
           'j_dim': JBLOCK_SIZE * POSES_PER_CONSTRAINT}

    lm_kernel = generatehessianDiagonalCode(name + '_lm', lm_param)
    if _PRINT_CODE:
        print hessian_code
    return op2.Kernel(hessian_code, name), lm_kernel


def generatehessianInitialCode(name):
    """ updates the first pose to make matrix non singular
        called:
        op2.par_loop(initial, initial_to_poses,
                     hessian((poses[op2.i[0]], poses[op2.i[0]]), op2.INC))
    """

    initial_code = """
    void %(name)s(double H[%(p_dim)d][%(p_dim)d], int i, int j) {
        //std::cout << "test" << std::endl;
        int idx1, idx2;
        for ( idx1 = 0; idx1 < %(p_dim)d; ++idx1 ) {
            for ( idx2 = 0; idx2 < %(p_dim)d; ++idx2 ) {
                H[idx1][idx2] *= 2.;
            }
        }
    }
    """ % {'name': name, 'p_dim': POSES_DIM}

    if _PRINT_CODE:
        print initial_code

    return op2.Kernel(initial_code, name)


def generatehessianDiagonalCode(name, lm_param):
    """ this updates the matrix via the lm_param value
        called as follows:
        op2.par_loop( diag_kernel, poses(1,1),
            hessian((poses_to_poses[p2.i[0]],
                         poses_to_poses[op2.i[0]]), op2.INC) )
    """
    diag_code = """
    void %(name)s(double H[%(p_dim)d][%(p_dim)d], int i, int j)
    {
        int idx = 0;
        for ( ; idx < %(p_dim)d; ++idx ) {
            H[idx][idx] *= %(lm_param)f;
        }
    }
    """ % {'name': name, 'p_dim': POSES_DIM, 'lm_param': (1 + lm_param)}

    if _PRINT_CODE:
        print diag_code

    return op2.Kernel(diag_code, name)


def generateUpdate(funcs):
    """ update the position using SE2 manifold updating
        how to call:
        op2.par_loop(update_kernel, poses,
                     dx(op2.IdentityMap, op2.READ),
                     x(op2.IdentityMap, op2.RW))
    """
    # code = ['x[%d] = dx[%d];'%(i, i) for i in xrange(POSES_DIM)]
    pose1_symbols = sympy.symbols(', '.join(['x[%d]' % i
                                  for i in xrange(POSES_DIM)]))
    pose2_symbols = sympy.symbols(', '.join(['dx[%d]' % i
                                  for i in xrange(POSES_DIM)]))

    code = []
    for i in xrange(POSES_DIM):
        c_code_lines = parse_return(generate_from_sympy(funcs[i](*(pose1_symbols + pose2_symbols)))).split('\n')
        for j, line in enumerate(c_code_lines):
            if (line[-1] != ';') and ('{' not in line and '}' not in line):
                c_code_lines[j] = 'x[%d] = %s;' % (i, line.strip())
        code.append('\n'.join(c_code_lines))

    update_code = """
    void update(double dx[%(p_dim)s], double x[%(p_dim)s]) {
        %(code)s
    }
    """ % {'p_dim': POSES_DIM, 'code': '\n'.join(code)}
    
    if _PRINT_CODE:
        print update_code
    return op2.Kernel(update_code, 'update')


def runBA():
    """ 2D bundle adjustment using PyOP2 and Sympy
        (for automatic kernel generation and differentiation)
    """

    if _PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    times = {'sse': [], 'estimate': [], 'error': [], 'jacobian': [],
             'rhs': [], 'lhs': [], 'solve': []}
    t0 = time.time()
    poses, constraints, initial, constraints_to_poses, constraints_to_constraints, poses_to_poses, initial_to_poses = preprocess()
    times['preprocess'] = (time.time() - t0)

    t0 = time.time()
    err, x, estimate, measurement, jacobian_blocks, omega_blocks, rhs_vec, dx, sse, sse_arr = setupData(poses, constraints)
    hessian = setuphessian(poses, constraints_to_poses)
    times['setup_data'] = (time.time() - t0)

    t0 = time.time()
    se2_funcs = [e_x, e_y, e_theta]
    se2_updates = [x_update, y_update, theta_update]
    estimate_kernel = generateEstimateCode('estimation', se2_funcs)
    error_kernel = generateErrorCode('errorize', se2_funcs)
    sse_kernel = generateSSECode('sum_of_squares')
    jacobian_kernel = generateJacobianCode('jacobian_block', se2_funcs)
    rhs_kernel = generateRHSCode('rhs')
    lhs_kernel, lm_kernel = generatehessianCode('lhs', 0.)
    update_kernel = generateUpdate(se2_updates)
    times['kernels'] = (time.time() - t0)

    if _DEBUG:
        # testKernel(x, constraints, constraints_to_poses)
        testKernel()

    solver = op2.Solver(linear_solver=SOLVER_TYPE,
                        preconditioner=PRECON,
                        relative_tolerance=1e-8,
                        maximum_iterations=1000,
                        error_on_nonconvergence=False)
    errors = []
    
    n = 0
    while n < MAX_ITER:

        t0 = time.time()
        rhs_vec.zero()
        dx.zero()
        sse.data = [0.0]
        hessian.zero()
        # estimate.zero()
        # err.zero()
        times['per_iter_setup'] = (time.time() - t0)

        t0 = time.time()
        op2.par_loop(estimate_kernel, constraints,
                     x(op2.READ, constraints_to_poses),
                     estimate(op2.WRITE))
        times['estimate'].append((time.time() - t0))

        t0 = time.time()
        op2.par_loop(error_kernel, constraints,
                     estimate(op2.READ),
                     measurement(op2.READ),
                     err(op2.WRITE))
        times['error'].append((time.time() - t0))

        t0 = time.time()

        op2.par_loop(sse_kernel, constraints,
                     err(op2.READ),
                     omega_blocks(op2.READ),
                     sse_arr(op2.WRITE),
                     sse(op2.INC))
        times['sse'].append((time.time() - t0))
        errors.append(sse.data)

        if _VERBOSE:
            print 'iteration %d: error: %f' % (n, sse.data)

        t0 = time.time()
        op2.par_loop(jacobian_kernel, constraints,
                     x(op2.READ, constraints_to_poses),
                     jacobian_blocks(op2.RW))
        times['jacobian'].append((time.time() - t0))

        t0 = time.time()
        op2.par_loop(rhs_kernel, constraints,
                     err(op2.READ),
                     jacobian_blocks(op2.READ),
                     omega_blocks(op2.READ),
                     rhs_vec(op2.INC, constraints_to_poses))
        times['rhs'].append((time.time() - t0))

        t0 = time.time()
        op2.par_loop(lhs_kernel, constraints,
                     jacobian_blocks(op2.READ),
                     omega_blocks(op2.READ),
                     hessian(op2.INC, (constraints_to_poses[op2.i[0]],
                                       constraints_to_poses[op2.i[0]])))
        
        # this sets up the initial condition
        hessian.handle[0:POSES_DIM, 0:POSES_DIM] *= 2.0

        op2.par_loop(lm_kernel, constraints,
                     hessian(op2.INC, (constraints_to_poses[op2.i[0]],
                                       constraints_to_poses[op2.i[0]])))

        times['lhs'].append((time.time() - t0))

        t0 = time.time()
        if _SAVE_DATA:
            np.save('error%d' % n, err.data)
            np.save('jacobians%d' % n, jacobian_blocks.data)
            np.save('omega', omega_blocks.data)
            np.save('hessian%d' % n, hessian.values)
            np.save('sse%d' % n, sse_arr.data)
            np.save('rhs%d' % n, rhs_vec.data)

        t0 = time.time()
        if _SOLVE:
            solver.solve(hessian, dx, rhs_vec)
            op2.par_loop(update_kernel, poses,
                         dx(op2.READ),
                         x(op2.RW))
        times['solve'].append((time.time() - t0))
        # print np.linalg.norm(dx.data)
        n += 1
        
    if _PROFILE:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.sort_stats('time').print_stats(0.1)

    summary = {k: np.mean(arr) for k, arr in times.iteritems()
               if k not in ['preprocess', 'kernels', 'setup_data']}
    pp.pprint(summary)
    print 'preprocess: %f generate kernels: %f setup data: %f' % \
          (times['preprocess'], times['kernels'], times['setup_data'])
    print 'total time per iteration: %f' % sum(summary.values())
    print 'total time: %f' % (MAX_ITER*sum(summary.values()))
    print 'initial error: %f' % errors[0]
    print 'final error: %f' % errors[-1]
    if _SAVE_RESULT:
        np.save('result', x.data)


# code generation tools
def generate_from_sympy(expr, derivative=False, wrt=None):
    """ given a SymPy expression 'expr', generates C-code for that expression
        or its derivative (if 'derivative' == True); if returning
        derivative, wrt should also be specified
    """
    global _expr_count
    _expr_count += 1
    name = 'expr_%d' % _expr_count
    if derivative and (wrt is not None):
        return codegen((name, sympy.diff(expr, wrt)), 'C', name)[0][1]
    else:
        return codegen((name, expr), 'C', name)[0][1]


def extract_args(code_string):
    """ find all arguments to C function and returns their names as a list """
    start = code_string.find('(') + 1
    end = code_string.find(')')
    return [s.split()[1] for s in code_string[start:end].split(',')]


def parse_return(code_string):
    """ parses a piece of C code which contains a function and
        returns the code immediately following the return statement
    """
    start = code_string.find('return ') + len('return ')
    end = code_string[start:].find(';')
    return code_string[start:start + end]


# definitions of measurement functions on different manifolds
def landmark_x(p_x, p_y, p_theta, l_x, l_y):
    return (p_x - l_x) * sympy.cos(p_theta) + (p_y - l_y) * sympy.sin(p_theta)


def landmark_y(p_x, p_y, p_theta, l_x, l_y):
    return -(p_x - l_x) * sympy.sin(p_theta) + (p_y - l_y) * sympy.cos(p_theta)


def r2_x(p_x, p_y, q_x, q_y):
    return q_x - p_x


def r2_y(p_x, p_y, q_x, q_y):
    return q_y - p_y


def e_x(p_x, p_y, p_theta, q_x, q_y, q_theta):
    """ symbolic x-component of error in SE2 """
    return (q_x - p_x) * sympy.cos(p_theta) + (q_y - p_y) * sympy.sin(p_theta)


def e_y(p_x, p_y, p_theta, q_x, q_y, q_theta):
    """ symbolic y-component of error in SE2 """
    return -(q_x - p_x) * sympy.sin(p_theta) + (q_y - p_y) * sympy.cos(p_theta)
    

def e_theta(p_x, p_y, p_theta, q_x, q_y, q_theta):
    """ symbolic theta-component of error in SE2 note that
        this should only be used for differentiation as Sympy
        cannot differentiate the Mod() function, so we do it a
        much more roundabout way
    """
    theta = q_theta - p_theta
    return Piecewise((theta, theta >= -np.pi and theta < np.pi),
                     (theta + 2*np.pi, theta < -np.pi),
                     (theta - 2*np.pi, theta >= np.pi))


def x_update(p_x, p_y, p_theta, dx, dy, dtheta):
    return p_x + dx * sympy.cos(p_theta) - dy * sympy.sin(p_theta)


def y_update(p_x, p_y, p_theta, dx, dy, dtheta):
    return p_y + dy * sympy.sin(p_theta) + dy * sympy.cos(p_theta)


def theta_update(p_x, p_y, p_theta, dx, dy, dtheta):
    theta = p_theta + dtheta
    return Piecewise((theta, theta >= -np.pi and theta < np.pi),
                     (theta + 2*np.pi, theta < -np.pi),
                     (theta - 2*np.pi, theta >= np.pi))


id_func = lambda x: x


# tools to generate matrix multiplication code
def dotproduct(first, st1, stride1, second, st2, stride2, num,
               idxFunc1=id_func, idxFunc2=id_func, prefix1='', prefix2=''):
    """ create the dot product of array 'first' starting at index st1 going
        by stride1 strides num times, with indices being re-mapped by idxFunc1;
        similar approach for 'second' array
    """
    ranges = zip(xrange(st1, st1 + stride1 * num, stride1),
                 xrange(st2, st2 + stride2 * num, stride2))
    return ' + '.join(['%s[%s%d]*%s[%s%d]' % (first, prefix1, idxFunc1(i),
                                              second, prefix2, idxFunc2(j))
                       for (i, j) in ranges])


def range_sum(N):
    return N * (N + 1) / 2


def sequence_sum(start, end):
    return range_sum(end) - range_sum(start)


def posdef_index(i, j, dim):
    """ gives the equivalent row index of the i, j matrix
        position if the matrix is positive-definite
    """
    if j >= i:
        thus_far = sequence_sum(dim-i, dim) if i > 0 else 0
        return thus_far + (j - i)
    else:
        thus_far = sequence_sum(dim - j, dim)
        return thus_far + (i - j)


def normal_to_posdef(idx, dim):
    """ given a normal C-style memory layout for a matrix,
        get back the equivalent positive-definite index
    """
    i = idx // dim
    j = idx - i * dim
    return posdef_index(i, j, dim)


if __name__ == '__main__':
    runBA()
