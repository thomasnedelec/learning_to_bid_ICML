import numpy as np

def compute_increasing_function_constraint_matrix_and_vector(coeffs_bidder_shading,
                                                             gradient_step_size):
    """
    :param knots_bidder:
    :param coeffs_bidder_shading:
    :return: matrix A and vector b such that Ax<=b in LP; x begin the rho coeffs
    """
    number_coeffs = len(coeffs_bidder_shading)
    A=np.zeros((number_coeffs - 1, number_coeffs))
    k = 0
    while (k < number_coeffs - 1):
        A[k,1:(k+2)]=1.0
        k += 1

    A_matrix = np.matrix(A)
    constraint_v1 = A_matrix.dot(coeffs_bidder_shading)
    constraint = constraint_v1/gradient_step_size

    return -A_matrix, constraint


def compute_less_than_id_constraint_matrix_and_vector(coeffs_bidder_shading,
                                                             bidder_knots,gradient_step_size):
    """
        :param coeffs_bidder_shading:
        :param bidder_knots:
        :return:
        """
    number_coeffs = len(coeffs_bidder_shading)
    A = np.zeros((number_coeffs - 1, number_coeffs))
    ineq_constraint_temp = np.zeros((number_coeffs - 1, 1))
    ineq_constraint = np.zeros((number_coeffs - 1, 1))

    A[0, 1] = 1.0
    k = 0
    while k < (number_coeffs - 2):
        A[k + 1, 0] = 1
        A[k + 1, 1] = bidder_knots[k]
        for j in range(0, k):
            A[k + 1, j] = max(bidder_knots[k] - bidder_knots[j], 0)  # this is for readability;
            # could actually take out the max since they are ordered
        k += 1

    ineq_A_matrix = np.matrix(A)
    temp = A @ coeffs_bidder_shading #matrix vector multiply
    ineq_constraint_temp[0] = 1 - coeffs_bidder_shading[1]
    temp_2 = coeffs_bidder_shading[2:] - temp[1:]
    ineq_constraint_temp[1:] = temp_2.reshape(number_coeffs - 2, 1) #just so it can be put into that vector...
    ineq_constraint = ineq_constraint_temp / gradient_step_size
    eq_constraint_vector=np.zeros((number_coeffs,1))
    eq_constraint_vector[0]=1.0
    eq_constraint_value = 0.0

    return {'ineq_mat': ineq_A_matrix, 'ineq_constraint': ineq_constraint,
            'eq_const_vec': eq_constraint_vector,'eq_constraint_value': eq_constraint_value}