"""
solver.py
DC-OPF Solver using Price Sensitive Demands

Fore more information about this method, refer to:
https://www2.econ.iastate.edu/tesfatsi/ABMAOPAMES.LT.pdf Page:50
and
https://www2.econ.iastate.edu/tesfatsi/DC-OPF.PriceSensitiveDemandBids.pdf

Converting to a fixed demand solver. Refering to:
http://www2.econ.iastate.edu/tesfatsi/DC-OPF.JSLT.pdf

Memoize cache using Joblib at end of file

"""

import os
import numpy as np
import typing
import tempfile

import joblib  # Using Joblib at the end of the file


from .logger import get_logger

""" Adding cache """
# memory = joblib.Memory(
#     tempfile.TemporaryDirectory(
#         prefix="solver_joblib_cache_",
#         dir=os.path.dirname(__file__)
#     ).name,
#     verbose=0,
#     compress=0,
#     bytes_limit=10_000,
# )
memory = joblib.Memory(
    os.path.join(os.path.dirname(__file__), "solver_joblib_memory"),
    verbose=0,
    compress=2,
    # bytes_limit=50_000,
)


@memory.cache
def solve(
        # Generators
        generator_a,
        generator_b,
        generator_locations,
        generator_commitement_minimum,
        generator_commitement_maximum,
        # Demands
        demand,
        demand_locations,
        # Lines
        line_start_nodes,
        line_end_nodes,
        line_reactances,
        line_capacity,
):
    # memory.reduce_size()
    tolerance = 0  # I am using this number to compare numebrs that are close to zero
    number_of_generators = len(generator_a)
    number_of_demands = len(demand)
    number_of_lines = len(line_start_nodes)
    number_of_nodes = int(max(  # Get the number of nodes using the maximum locations anywhere
        max(generator_locations),
        max(demand_locations),
        max(line_start_nodes),
        max(line_end_nodes)
    ) + 1)
    # Note: I -> Number of Generators
    #       J -> Number of LSEs
    #       K -> Number of Nodes
    """ Objective Function Representation """
    # f(x) = 1/2 * x.T * G * x + a.T * x
    nodes_zeros = np.zeros(number_of_nodes - 1)
    a = np.hstack((generator_a, nodes_zeros))
    # a = a.reshape(-1, 1)  # Transpose the a
    U = np.diag(2 * generator_b)
    line_adjacency_matrix = np.zeros((number_of_lines, number_of_nodes))
    for node_i in range(number_of_nodes):
        for line_i in range(number_of_lines):
            if line_start_nodes[line_i] == node_i:
                line_adjacency_matrix[line_i, node_i] = +1
            elif line_end_nodes[line_i] == node_i:
                line_adjacency_matrix[line_i, node_i] = -1
    node_penalty = 0.05  # Notation as PI (π) symbol
    W_rr = 2 * node_penalty * np.dot(line_adjacency_matrix.T, line_adjacency_matrix)[1:, 1:]
    G = diag_block(U, W_rr)

    """ Equality Constraints Representation """
    # C^T_eq * x = beq
    II = np.zeros((number_of_nodes, number_of_generators))
    for row_i in range(number_of_nodes):
        for col_i in range(number_of_generators):
            if generator_locations[col_i] == row_i:
                II[row_i, col_i] = 1
    JJ = np.zeros((number_of_nodes, number_of_demands))
    for row_i in range(number_of_nodes):
        for col_i in range(number_of_demands):
            if demand_locations[col_i] == row_i:
                JJ[row_i, col_i] = 1
    B_r = np.dot((1/line_reactances) * line_adjacency_matrix.T, line_adjacency_matrix)[1:]
    Ceq = np.hstack((II, -B_r.T))
    Ceq = Ceq.T
    beq = np.dot(JJ, demand.T)
    # beq = beq.reshape(-1, 1)

    """ Inequality Constraints """
    A_r = line_adjacency_matrix[:, 1:]  # Reduced Adjacency Matrix
    Z = np.diag(1 / line_reactances)
    ZA_r = np.dot(Z, A_r)
    Ciq = np.vstack((
        np.hstack((
            np.zeros((number_of_lines, number_of_generators)),
            # np.zeros((number_of_lines, number_of_demands)),
            ZA_r
        )),
        np.hstack((
            np.zeros((number_of_lines, number_of_generators)),
            # np.zeros((number_of_lines, number_of_demands)),
            -ZA_r
        )),
        np.hstack((
            np.identity(number_of_generators),
            # np.zeros((number_of_generators, number_of_demands)),
            np.zeros((number_of_generators, number_of_nodes - 1))
        )),
        np.hstack((
            - np.identity(number_of_generators),
            # np.zeros((number_of_generators, number_of_demands)),
            np.zeros((number_of_generators, number_of_nodes - 1))
        )),
        # np.hstack([
        #     np.zeros((number_of_demands, number_of_generators)),
        #     np.identity(number_of_demands),
        #     np.zeros((number_of_demands, number_of_nodes - 1))
        # ]),
        # np.hstack([
        #     np.zeros((number_of_demands, number_of_generators)),
        #     - np.identity(number_of_demands),
        #     np.zeros((number_of_demands, number_of_nodes - 1))
        # ])
    ))
    Ciq = Ciq.T  # Transpose the Ciq
    biq = np.hstack((
        -line_capacity,
        -line_capacity,
        generator_commitement_minimum,
        - generator_commitement_maximum,
        # demand_sensitive_minimum,
        # - demand_sensitive_maximum
    ))
    # biq = biq.reshape(-1, 1)

    """ Compute additional parameters for the optimization """
    n = a.size
    meq = beq.size
    miq = biq.size
    m = meq + miq
    C = np.hstack((Ceq, Ciq))
    b = np.hstack((beq, biq))
    scp = 1
    a0 = 0.0
    is_infeasible = False
    is_feasible_and_optimum = False
    L = np.linalg.cholesky(G)
    N = np.zeros((n, np.minimum(m, n)))
    num_drop = 0
    num_add = 0
    num_iter = 0

    """ Find Unconstrained minimum """
    x = - np.dot(np.linalg.inv(G), a)
    f = 0.5 * np.dot(a, x)
    H = np.linalg.inv(G)  # Do I need to make a deepcopy?
    A = np.zeros(np.minimum(m, n))
    q = 0
    # f_hist = []  # DEBUG
    # f_hist.append(0.5 * np.linalg.multi_dot([x, G, x]) + np.dot(a, x))  # DEBUG

    """ Find Equality Constrained Minimum """
    GCeq = np.vstack((
        np.hstack((
            G, 
            -Ceq
            )),
        np.hstack((
            -Ceq.T, 
            np.zeros((Ceq.shape[1], Ceq.shape[1]))
            )),
    ))
    abeq = np.hstack((-a, -beq))
    ecsol = np.dot(np.linalg.inv(GCeq), abeq)
    x = ecsol[0:n].copy()
    u = ecsol[n:n+meq].copy()
    f = 0.5 * np.dot(np.dot(x, G), x) + np.dot(a, x)
    x = correct_rounding_error_iterable(x)
    f = correct_rounding_error_scalar(f)
    if q > 0:
        u = correct_rounding_error_iterable(u)
    for i in range(0, meq):
        A[i] = i
    q = meq
    N[:n, :meq] = Ceq

    # f_hist.append(0.5 * np.linalg.multi_dot([x, G, x]) + np.dot(a, x))  # DEBUG

    """ Update H and Nstar """
    B = np.dot(np.linalg.inv(L), N[:n, :q])
    Q1, R = np.linalg.qr(B)
    J1 = np.linalg.solve(L.T, Q1)
    # Q2Q2T = - np.dot(Q1, Q1.T)
    Q2Q2T = np.identity(n) - np.dot(Q1, Q1.T)
    H = np.dot(
        np.dot(np.linalg.inv(L.T), Q2Q2T), 
        np.linalg.inv(L)
        )
    Nstar = np.linalg.solve(R, J1.T)

    """ Choose violated Constraint """
    siq = np.dot(Ciq.T, x) - biq
    siq = correct_rounding_error_iterable(siq)
    nvc = (siq < -tolerance).sum()   # TODO: Check if tolerance needed
    if nvc == 0:
        is_feasible_and_optimum = True
        # print("No violated constraints")
    else:
        V = np.array([i for i in range(siq.size) if siq[i] < -tolerance])  # TODO: Check if tolerance needed
        # Note to self: the strategy int `scp` is always 1
        temp = siq.min()
        p = meq + np.argmin(siq)
        nplus = C[:, p]
        if q == 0:
            uplus = np.empty(1)
            u = np.zeros(1)
        else:
            uplus = np.append(u, np.empty(1))
        # print("Choose violated Constraint: nvc = ", nvc)

    """ Start the iterations """
    num_iter = 0
    while (not is_feasible_and_optimum) and (not is_infeasible):
        num_iter = num_iter + 1
        """ Determine Step Direction """
        z = np.dot(H, nplus)
        z = correct_rounding_error_iterable(z)
        if q > 0:
            r = np.dot(Nstar, nplus)
            r = correct_rounding_error_iterable(r)
        else:
            r = None
        """ Compute Step Length """  # Partial then Full then t = min(t1, t2)
        """ Compute Partial Step Length"""
        if q == 0:
            t1 = 1e15  # A huge number??
        else:
            npe = 0
            for i in range(0, q - meq):
                if r[meq + i] > tolerance:  # Tolerance?
                    npe = npe + 1
            if npe == 0:
                t1 = 1e15
            else:
                temp_min = 0
                ctbd = 0
                t1 = 1e15
                for j in range(0, q - meq):
                    if r[meq + j] > tolerance:  # Tolerance?
                        temp_min = uplus[meq + j] / r[meq + j]
                        if temp_min < t1:
                            t1 = temp_min
                            ctbd = meq + j
                k = A[ctbd]

        """ Compute Full Step Length """
        if np.dot(z, z) == 0:
            t2 = 1e15
        else:
            sp = np.dot(nplus, x) - b[p]
            t2 = -sp / np.dot(z, nplus)

        """ End of partial and full step length """
        t = np.minimum(t1, t2)
        # print("t = min(t1, t2) = min(partial step, full step) = min(", t1, ", ", t2, ") = ",  t)

        """ Determine new S-pair and Take Step """
        if t == 1e15:
            is_infeasible = True
            get_logger().error("Problem is infeasible!")
            raise Exception("Problem is infeasible!")
        elif t2 == 1e15:
            is_full_step = False
            rplus = np.append(-r, 1.0)
            uplus = uplus + rplus * t
            """ Drop Zero Multiplier Corresponding to constraint k """
            # print("Drop Zero Multiplier Corresponding to constraint k = ", k)
            for i in range(meq, A.size):
                if A[i] == k:
                    for j in range(i, uplus.size - 1):
                        uplus[j] = uplus[j+1]
                    utemp = uplus[0: uplus.size - 1]
                    uplus = utemp.copy()
                    break
            """ Drop Constraint K """
            # print("Drop Constraint k = ", k)
            num_drop = num_drop + 1
            q = q - 1
            for i in range(meq, A.size):
                if A[i] == k:
                    for j in range(i, q):
                        A[j] = A[j + 1]
                        N[:, j] = N[:, j + 1]
                    A[q] = 0
                    N[:, q] = 0
                    break
            """ Update H and Nstar """
            B = np.dot(np.linalg.inv(L), N[:n, :q])
            Q1, R = np.linalg.qr(B)
            J1 = np.linalg.solve(L.T, Q1)
            # Q2Q2T = - np.dot(Q1, Q1.T)
            Q2Q2T = np.identity(n) - np.dot(Q1, Q1.T)
            H = np.dot(
                np.dot(np.linalg.inv(L.T), Q2Q2T),
                np.linalg.inv(L)
                )
            Nstar = np.linalg.solve(R, J1.T)

        else:
            x = x + z * t
            f = f + t * np.dot(z, nplus) * (0.5 * t + uplus[uplus.size - 1])
            rplus = np.append(-r, 1.0)
            uplus = uplus + rplus * t  # not sure
            if t2 <= t1:
                is_full_step = True
                u = uplus.copy()
                """ Add Constraint P """
                # print("Add Constraint P = ", p)
                num_add = num_add + 1
                A[q] = p
                N[:, q] = C[:, p]
                q = q + 1
                """ Update H and Nstar """
                B = np.dot(np.linalg.inv(L), N[:n, :q])
                Q1, R = np.linalg.qr(B)
                J1 = np.linalg.solve(L.T, Q1)
                # Q2Q2T = - np.dot(Q1, Q1.T)
                Q2Q2T = np.identity(n) - np.dot(Q1, Q1.T)
                H = np.dot(
                    np.dot(np.linalg.inv(L.T), Q2Q2T),
                    np.linalg.inv(L)
                    )
                Nstar = np.linalg.solve(R, J1.T)

            else:
                is_full_step = False
                """ Drop Zero Multiplier Corresponding to constraint k """
                # print("Drop Zero Multiplier Corresponding to constraint k = ", k)
                for i in range(meq, A.size):
                    if A[i] == k:
                        for j in range(i, uplus.size - 1):
                            uplus[j] = uplus[j + 1]
                        utemp = uplus[0: uplus.size - 1]
                        uplus = utemp.copy()
                        break
                """ Drop Constraint K """
                # print("Drop Constraint K = ", k)
                num_drop = num_drop + 1
                q = q - 1
                for i in range(meq, A.size):
                    if A[i] == k:
                        for j in range(i, q):
                            A[j] = A[j + 1]
                            N[:, j] = N[:, j + 1]
                        A[q] = 0
                        N[:, q] = 0
                        break
                """ Update H and Nstar """
                B = np.dot(np.linalg.inv(L), N[:n, :q])
                Q1, R = np.linalg.qr(B)
                J1 = np.linalg.solve(L.T, Q1)
                # Q2Q2T = - np.dot(Q1, Q1.T)
                Q2Q2T = np.identity(n) - np.dot(Q1, Q1.T)
                H = np.dot(
                    np.dot(np.linalg.inv(L.T), Q2Q2T),
                    np.linalg.inv(L)
                    )
                Nstar = np.linalg.solve(R, J1.T)

        if is_full_step:
            """ Choose violated Constraint """
            siq = np.dot(Ciq.T, x) - biq
            siq = correct_rounding_error_iterable(siq)
            nvc = (siq < -tolerance).sum()  # TODO: Check if tolerance needed
            if nvc == 0:
                is_feasible_and_optimum = True
                # print("No violated constraints")
            else:
                V = np.array([i for i in range(siq.size) if siq[i] < -tolerance])  # TODO: Check if tolerance needed
                # Note to self: the strategy int `scp` is always 1
                temp = siq.min()
                p = meq + np.argmin(siq)
                nplus = C[:, p]
                if q == 0:
                    uplus = np.empty(1)
                    u = np.zeros(1)
                else:
                    uplus = np.append(u, np.empty(1))
        # f_hist.append(0.5 * np.linalg.multi_dot([x, G, x]) + np.dot(a, x))  # DEBUG

    x = correct_rounding_error_iterable(x)
    f = correct_rounding_error_scalar(f)
    if q > 0:
        u = correct_rounding_error_iterable(u)

    solution_commitement = x[0:number_of_generators]
    # solution_sensitive_demand =np.round(x[number_of_generators: number_of_generators + number_of_demands], 4)
    voltage_angles = x[number_of_generators:]
    solution_lmp = u[0:meq]

    return {
        'Commitements': solution_commitement,
        # "Sensitive Demands": solution_sensitive_demand,
        "Voltage Angles": voltage_angles,
        "LMPs": solution_lmp,
    }


def diag_block(arr1, arr2):
    arr = np.vstack((
        np.hstack((
            arr1, 
            np.zeros((arr1.shape[0], arr2.shape[1]))
            )),
        np.hstack((
            np.zeros((arr2.shape[0], arr1.shape[1])),
            arr2
            ))
    ))
    return arr


def correct_rounding_error_iterable(x: np.ndarray) -> np.ndarray:
    for i in range(x.size):
        if np.absolute(x[i] - np.round(x[i])) < 1e-9:
            x[i] = np.round(x[i])
    return x

def correct_rounding_error_scalar(x: typing.Union[int, float]):
    if np.absolute(x - np.round(x)) < 1e-9:
        x = np.round(x)
    return x
