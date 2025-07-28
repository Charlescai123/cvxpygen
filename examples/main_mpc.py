
import cvxpy as cp
from cvxpygen import cpg
import importlib
import numpy as np
import time
import sys

from MPC_code.cpg_solver import cpg_solve
import numpy as np
import pickle
import time

if __name__ == "__main__":

    # define dimensions
    H, n, m = 10, 6, 3

    # define variables
    U = cp.Variable((m, H), name='U')
    X = cp.Variable((n, H + 1), name='X')

    # define parameters
    Psqrt = cp.Parameter((n, n), name='Psqrt')
    Qsqrt = cp.Parameter((n, n), name='Qsqrt')
    Rsqrt = cp.Parameter((m, m), name='Rsqrt')
    A = cp.Parameter((n, n), name='A')
    B = cp.Parameter((n, m), name='B')
    x_init = cp.Parameter(n, name='x_init')

    # define objective
    objective = cp.Minimize(
        cp.sum_squares(Psqrt @ X[:, H]) + cp.sum_squares(Qsqrt @ X[:, :H]) + cp.sum_squares(Rsqrt @ U))

    # define constraints
    constraints = [X[:, 1:] == A @ X[:, :H] + B @ U,
                   cp.abs(U) <= 1,
                   X[:, 0] == x_init]

    # define problem
    problem = cp.Problem(objective, constraints)

    # continuous-time dynmaics
    A_cont = np.concatenate((np.array([[0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]]),
                             np.zeros((3, 6))), axis=0)
    mass = 1
    B_cont = np.concatenate((np.zeros((3, 3)),
                             (1 / mass) * np.diag(np.ones(3))), axis=0)

    # discrete-time dynamics
    td = 0.1
    A.value = np.eye(n) + td * A_cont
    B.value = td * B_cont

    # cost
    Psqrt.value = np.eye(n)
    Qsqrt.value = np.eye(n)
    Rsqrt.value = np.sqrt(0.1) * np.eye(m)

    # measurement
    x_init.value = np.array([2, 2, 2, -1, -1, 1])

    val = problem.solve()

    # module = importlib.import_module(f'MPC_code.cpg_solver')
    # cpg_solve = getattr(module, 'cpg_solve')
    # problem.register_solve('CPG', cpg_solve)
    cpg.generate_code(problem, code_dir='MPC_code', solver='ECOS')

    # load the serialized problem formulation
    with open('MPC_code/problem.pickle', 'rb') as f:
        prob = pickle.load(f)

    # assign parameter values
    prob.param_dict['A'].value = np.eye(n) + td * A_cont
    prob.param_dict['B'].value = td * B_cont
    prob.param_dict['Psqrt'].value = np.eye(n)
    prob.param_dict['Qsqrt'].value = np.eye(n)
    prob.param_dict['Rsqrt'].value = np.sqrt(0.1) * np.eye(m)
    prob.param_dict['x_init'].value = np.array([2, 2, 2, -1, -1, 1])

    # solve problem conventionally
    t0 = time.time()
    # CVXPY chooses eps_abs=eps_rel=1e-5, max_iter=10000, polish=True by default,
    # however, we choose the OSQP default values here, as they are used for code generation as well
    val = prob.solve(eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polish=False)
    t1 = time.time()
    print('\nCVXPY\nSolve time: %.3f ms' % (1000 * (t1 - t0)))
    print('Objective function value: %.6f\n' % val)

    # solve problem with C code via python wrapper
    prob.register_solve('CPG', cpg_solve)
    t0 = time.time()
    val = prob.solve(method='CPG')
    t1 = time.time()
    print('\nCVXPYgen\nSolve time: %.3f ms' % (1000 * (t1 - t0)))
    print('Objective function value: %.6f\n' % val)
