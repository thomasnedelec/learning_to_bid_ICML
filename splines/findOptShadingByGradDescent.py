# general functions
# one strategic bidder
import numpy as np
import scipy
import warnings
import scipy.integrate as integrate
from scipy.optimize import linprog
import math

import sys
import os
#sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..'))
from computeFuncConstraintsInLPForm import *

####FIRST I CONSIDER POPULATION VERSIONS OF THE PROBLEM



def compute_x_0_from_v_value(compute_v_value_after_shading,test_vals=[0.0,1.0],my_tol=10**-4,assume_v_val_inc=False):
    """
    note: this sorts of assume an increasing v_value... or that the reserve value is at min of x_0 for which the virtualized bid is
    positive...
    also there is a problem in using it as is: this gives the smallest number so that min_val=beta^{-1}(x) is such that
    psi_B(min_val)>0; but we really need the smallest x... hence the function below
    :param compute_v_value_after_shading:
    :param test_vals:
    :param my_tol:
    :return:
    """
    if assume_v_val_inc is True:
        rand_init = 1.0 #np.random.rand(1,1)
        current_val = rand_init * test_vals[1]
        lower_bound=test_vals[0]
        upper_bound=test_vals[1]
        while upper_bound - lower_bound > my_tol:
            if compute_v_value_after_shading(current_val)>0:
                upper_bound=current_val
                current_val = (upper_bound+lower_bound)/2.0
            else:
                lower_bound=current_val
                current_val = (upper_bound + lower_bound) / 2.0
        return current_val
    else:
        grid = np.linspace(test_vals[0],test_vals[1],1001)
        index = 0
        max_index=len(grid)
        while index<max_index:
            if compute_v_value_after_shading(grid[index]) < 0.0:
                index += 1
            else:
                break;
        min_val = grid[index]

        # possible_values = grid[compute_v_value_after_shading(grid) >= 0.0]
        # min_val = possible_values[0]
        return min_val


def compute_x_0_from_v_value_brute_force(virtual_value_after_shading,
                                         bidder_shading,
                                         cdf_opponents,
                                         density_bidder,
                                        test_vals=[0.0,1.0],
                                        choose_smallest=True
):
    """
    this is stupid because it computes way too many integrals but it's just to debug stuff
    :param virtual_value_after_shading:
    :param bidder_shading:
    :param cdf_opponents:
    :param density_bidder:
    :param test_vals:
    :return:
    """
    upper_bound = test_vals[1]
    grid_size = 51
    grid = np.linspace(test_vals[0],test_vals[1],grid_size)
    index = 0
    tol = 0.006
    max_index=len(grid)
    revenue_seller = np.zeros(max_index)
    while index<max_index:
        revenue_seller[index] = compute_current_revenue_seller(cdf_opponents,bidder_shading,virtual_value_after_shading,
                                   density_bidder,grid[index],upper_bound)
        index += 1

    index_best_x = np.argmax(revenue_seller)
    #print(f"revenue_seller:{revenue_seller}")
    #print(f"revenuemax:{revenue_seller[index_best_x]}")
    #if choose_smallest:
        #print([i for i in range(max_index) if math.isclose(revenue_seller[i], revenue_seller[index_best_x], rel_tol=0.0, abs_tol=tol)])
    #    index_smallest_x = next(i for i in range(max_index) if math.isclose(revenue_seller[i], revenue_seller[index_best_x], rel_tol=0.0, abs_tol=tol))
    index_smallest_x = index_best_x
    return grid[index_smallest_x] , revenue_seller
    #return grid[index_best_x]


def compute_x_0_from_v_value_brute_force_smarter(virtual_value_after_shading,
                                         bidder_shading,
                                         cdf_opponents,
                                         density_bidder,
                                        test_vals=[0.0,1.0]
):
    """
    this is stupid because it computes way too many integrals but it's just to debug stuff
    :param virtual_value_after_shading:
    :param bidder_shading:
    :param cdf_opponents:
    :param density_bidder:
    :param test_vals:
    :return:
    """
    upper_bound = test_vals[1]
    grid_size = 101
    K = 5 #number of things to try
    grid = np.linspace(test_vals[0],test_vals[1],grid_size)

    temp_0 = [virtual_value_after_shading(grid_points) for grid_points in grid]
    temp = np.absolute(temp_0)
    idx = np.argpartition(temp, K) #returns the indices of the K smallest elements of temp
    interesting_grid = grid[idx]


    max_index=len(interesting_grid)
    revenue_seller = np.zeros(max_index)

    for index, int_point in enumerate(interesting_grid):
        revenue_seller[index] = compute_current_revenue_seller(cdf_opponents,bidder_shading,virtual_value_after_shading,
                                   density_bidder,int_point,upper_bound)

    index_best_x = np.argmax(revenue_seller)

    #return grid[index_best_x] , revenue_seller
    return interesting_grid[index_best_x]




def compute_reserve_value_from_0_of_v_value(x_0_from_v_value,bidder_shading,test_vals=[0.0,1.0],my_tol=10**-4):
    """
    note: I assume bidder_shading is increasing
    :param x_0_from_v_value:
    :param bidder_shading:
    :param test_vals:
    :param my_tol:
    :return:
    """
    rand_init = 1.0  # np.random.rand(1,1)
    current_val = rand_init * test_vals[1]
    lower_bound = test_vals[0]
    upper_bound = test_vals[1]
    while upper_bound - lower_bound > my_tol:
        if bidder_shading(current_val) > x_0_from_v_value:
            upper_bound = current_val
            current_val = (upper_bound + lower_bound) / 2.0
        else:
            lower_bound = current_val
            current_val = (upper_bound + lower_bound) / 2.0
    return current_val



def compute_key_function(density_opponents, bidder_shading, virtual_value_after_shading,
                         density_bidder, basis_function, x_value, x_0_value, assume_increasing_v_value=False):
    """

    :param density_opponents:
    :param bidder_shading:
    :param virtual_value_after_shading:
    :param density_bidder:
    :param basis_function:
    :param x_value:
    :param x_0_value:
    :param assume_increasing_v_value:
    :return:
    """

    if assume_increasing_v_value is True:
        indicator = (x_value >= x_0_value)
    else:
        indicator = (virtual_value_after_shading(x_value) >= 0.0)

    if (indicator == 0):
        temp=0.0
    else:
        temp_1 = density_opponents(bidder_shading(x_value)) * (x_value - bidder_shading(x_value))
        temp = temp_1 * basis_function(x_value) * density_bidder(x_value)
    return temp


def compute_current_revenue_bidder(cdf_opponents,bidder_shading,virtual_value_after_shading,
                                   density_bidder,x_0_value,upper_bound=1.0):
    """
    computes the current revenue of the bidder
    :param cdf_opponents:
    :param bidder_shading:
    :param virtual_value_after_shading:
    :param density_bidder:
    :param x_0_value:
    :param upper_bound:
    :return:
    """
    print(x_0_value)
    temp_21 = integrate.quad(
        lambda x: (x-virtual_value_after_shading(x))*cdf_opponents(bidder_shading(x))*density_bidder(x),x_0_value,upper_bound)
    temp = temp_21[0]
    return temp


def compute_current_revenue_seller(cdf_opponents,bidder_shading,virtual_value_after_shading,
                                   density_bidder,x_0_value,upper_bound=1.0):
    """
    computes the current revenue of the bidder
    :param cdf_opponents:
    :param bidder_shading:
    :param virtual_value_after_shading:
    :param density_bidder:
    :param x_0_value:
    :param upper_bound:
    :return:
    """
    temp_21 = integrate.quad(
        lambda x: virtual_value_after_shading(x)*cdf_opponents(bidder_shading(x))*density_bidder(x),x_0_value,upper_bound)
    temp = temp_21[0]
    return temp


def compute_grad_step_bidder_payoff_direction_basis_function_no_reserve_move(density_opponents,
                                                                             bidder_shading,
                                                                             virtual_value_after_shading,
                                                                             density_bidder,
                                                                             basis_function,
                                                                             x_0_value,
                                                                             assume_v_val_inc=False,
                                                                             lower_bound=0.0,
                                                                             upper_bound=1.0):
    """
    computes a grad step in direction of shading function; only first part of equation
    as in this function we don't take into account the reserve price move
    :param density_opponents:
    :param bidder_shading:
    :param virtual_value_after_shading:
    :param basis_function:
    :return:
    """
    temp_21 = integrate.quad(
        lambda x: compute_key_function(density_opponents, bidder_shading, virtual_value_after_shading,
                                       density_bidder, basis_function, x,x_0_value,assume_v_val_inc),lower_bound, upper_bound)
    temp = temp_21[0]
    # print("key integral", temp)
    # print("x_0_value",x_0_value)
    return temp


def compute_grad_step_bidder_payoff_direction_basis_function_only_reserve_move(density_opponents, cdf_opponents,
                                                                               bidder_shading, bidder_shading_derivative,
                                                                               knots_bidder,
                                                                               virtual_value_after_shading,
                                                                               density_bidder, cdf_bidder,
                                                                               basis_function,
                                                                               deriv_basis_function,
                                                                               x_0):
    """
    computes the impact of the reserve price move on the value of the bidder
    :param density_opponents:
    :param cdf_opponents:
    :param bidder_shading:
    :param virtual_value_after_shading:
    :param density_bidder:
    :param cdf_bidder:
    :param basis_function:
    :param deriv_basis_function:
    :param x_0:
    :return:s
    """
    #print(f"knots_bidder:{knots_bidder}")
    p_win_at_shaded_value = cdf_opponents(bidder_shading(x_0))
    psi_like_function = x_0 * density_bidder(x_0)/(2*bidder_shading_derivative(x_0)) - (1 - cdf_bidder(x_0))
    first_term = p_win_at_shaded_value * psi_like_function * basis_function(x_0)
    #print(f"x_0:{x_0}")
    #print(bidder_shading_derivative)
    #print(f"biddershadingderviative(x_0){bidder_shading_derivative(x_0)}")
    second_term = deriv_basis_function(x_0) * x_0 * (1 - cdf_bidder(x_0)) /(2*bidder_shading_derivative(x_0))*p_win_at_shaded_value
    infinitesimal_move = first_term - second_term


    return infinitesimal_move


###first order splines

def compute_grad_step_bidder_payoff_direction_basis_function(density_opponents, cdf_opponents,
                                                             bidder_shading, bidder_shading_derivative,knots_bidder,
                                                             virtual_value_after_shading,
                                                             density_bidder, cdf_bidder,
                                                             basis_function, deriv_basis_function,
                                                             x_0, assume_v_val_inc=False,
                                                             lower_bound=0.0, upper_bound=1.0):
    """

    :param density_opponents:
    :param cdf_opponents:
    :param bidder_shading:
    :param virtual_value_after_shading:
    :param density_bidder:
    :param cdf_bidder:
    :param basis_function:
    :param deriv_basis_function:
    :param x_0:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    second_term = compute_grad_step_bidder_payoff_direction_basis_function_only_reserve_move(density_opponents,
                                                                                             cdf_opponents,
                                                                                             bidder_shading,
                                                                                             bidder_shading_derivative,
                                                                                             knots_bidder,
                                                                                             virtual_value_after_shading,
                                                                                             density_bidder, cdf_bidder,
                                                                                             basis_function,
                                                                                             deriv_basis_function,
                                                                                             x_0)

    first_term = compute_grad_step_bidder_payoff_direction_basis_function_no_reserve_move(density_opponents,
                                                                                          bidder_shading,
                                                                                          virtual_value_after_shading,
                                                                                          density_bidder,
                                                                                          basis_function,
                                                                                          x_0,
                                                                                          assume_v_val_inc,
                                                                                          lower_bound,
                                                                                          upper_bound)

    return first_term + second_term


def compute_bidder_shading(knots_bidder, coeffs_bidder_shading):
    def this_function(x): return np.dot(coeffs_bidder_shading,
                                        np.append([1.0, x], [max(0.0, x - knot) for knot in knots_bidder]))

    return this_function


def compute_derivative_bidder_shading(knots_bidder, coeffs_bidder_shading):
    # this_function = lambda x: np.dot(coeffs_bidder_shading, np.append([0, 1], [( x>= knot) for knot in knots_bidder]))
    def this_function(x): return np.dot(coeffs_bidder_shading,
                                        np.append([0, 1], [(x >= knot) for knot in knots_bidder]))

    return this_function

def compute_v_value_after_shading(knots_bidder, coeffs_bidder_shading, bidder_pdf, bidder_cdf):
    """
    implements the differential equation results we had a long time ago...
    note that we replace the subderivative at the knots by a value of the subderivative...
    :param knots_bidder:
    :param coeffs_bidder_shading:
    :param bidder_pdf:
    :param bidder_cdf:
    :return:
    """
    this_shading = compute_bidder_shading(knots_bidder, coeffs_bidder_shading)
    this_der_shading = compute_derivative_bidder_shading(knots_bidder, coeffs_bidder_shading)
    def v_value_bidder(x): return  (1.0 - bidder_cdf(x)) / bidder_pdf(x)

    def this_function(x): return this_shading(x) - this_der_shading(x) * v_value_bidder(x)

    return this_function




def compute_objective_vector_first_order_splines(density_opponents, cdf_opponents,
                                                 knots_bidder,
                                                 coeffs_bidder_shading,
                                                 density_bidder, cdf_bidder,
                                                 x_0,
                                                 assume_v_val_inc=False,
                                                 lower_bound=0.0, upper_bound=1.0):
    """
    :param density_opponents:
    :param cdf_opponents:
    :param knots_bidder:
    :param coeffs_bidder_shading:
    :param virtual_value_after_shading:
    :param density_bidder:
    :param cdf_bidder:
    :param x_0:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    objective_vector = np.zeros(len(coeffs_bidder_shading))
    # this is really poorly coded...

    this_shading_function = compute_bidder_shading(knots_bidder, coeffs_bidder_shading)
    virtual_value_after_shading = compute_v_value_after_shading(knots_bidder,
                                                                coeffs_bidder_shading, density_bidder, cdf_bidder)
    bidder_shading_derivative = compute_derivative_bidder_shading(knots_bidder, coeffs_bidder_shading)

    for index, this_coeff in enumerate(coeffs_bidder_shading):
        if (index >= 2):
            this_knot = knots_bidder[index - 2]
            temp = compute_grad_step_bidder_payoff_direction_basis_function(density_opponents, cdf_opponents,
                                                                            this_shading_function,
                                                                            bidder_shading_derivative,
                                                                            knots_bidder,
                                                                            virtual_value_after_shading,
                                                                            density_bidder, cdf_bidder,
                                                                            lambda x: max(0.0, x - this_knot),
                                                                            lambda x: (x >= this_knot),
                                                                            x_0,
                                                                            assume_v_val_inc,
                                                                            lower_bound, upper_bound)
            objective_vector[index] = temp
        elif (index == 0):
            temp = compute_grad_step_bidder_payoff_direction_basis_function(density_opponents, cdf_opponents,
                                                                            this_shading_function,bidder_shading_derivative,
                                                                            knots_bidder,
                                                                            virtual_value_after_shading,
                                                                            density_bidder, cdf_bidder,
                                                                            lambda x: 1.0,
                                                                            lambda x: 0.0,
                                                                            x_0,
                                                                            assume_v_val_inc,
                                                                            lower_bound, upper_bound)
            objective_vector[index] = temp
        else:
            temp = compute_grad_step_bidder_payoff_direction_basis_function(density_opponents, cdf_opponents,
                                                                            this_shading_function,bidder_shading_derivative,
                                                                            knots_bidder,virtual_value_after_shading,
                                                                            density_bidder, cdf_bidder,
                                                                            lambda x: x,
                                                                            lambda x: 1.0,
                                                                            x_0,
                                                                            assume_v_val_inc,
                                                                            lower_bound, upper_bound)
            objective_vector[index] = temp

    #print(objective_vector)
    return objective_vector





def define_and_solve_one_gradient_step(density_opponents, cdf_opponents,
                                                 knots_bidder,
                                                 coeffs_bidder_shading,
                                                 density_bidder, cdf_bidder,
                                                 x_0, assume_v_val_inc=False,
                                       lower_bound=0.0, upper_bound=1.0,
                                        gradient_step_size=10**-2,
                                        abs_max_size_move=1.0,
                                        less_than_id_constraint=False,
                                       monotonicity_constraint=False):
    """

    :param density_opponents:
    :param cdf_opponents:
    :param knots_bidder:
    :param coeffs_bidder_shading:
    :param density_bidder:
    :param cdf_bidder:
    :param x_0:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    reward_vector=compute_objective_vector_first_order_splines(density_opponents, cdf_opponents,
                                                 knots_bidder,
                                                 coeffs_bidder_shading,
                                                 density_bidder, cdf_bidder,
                                                 x_0,assume_v_val_inc,
                                                lower_bound, upper_bound)

    if monotonicity_constraint is False:
        new_coeffs = coeffs_bidder_shading + gradient_step_size * reward_vector
    else:
        monotonicity_c_matrix,monotonicity_c_vec = compute_increasing_function_constraint_matrix_and_vector(coeffs_bidder_shading,
                                                             gradient_step_size)
    # print(monotonicity_c_matrix)
    # print(monotonicity_c_vec)

        if less_than_id_constraint is False:
             update_move = scipy.optimize.linprog(-gradient_step_size*reward_vector, A_ub=monotonicity_c_matrix, b_ub=monotonicity_c_vec,
                                A_eq=None, b_eq=None, bounds=(-abs_max_size_move,abs_max_size_move), method='simplex',
                                callback=None, options=None)
         #print("update move", update_move.x)
        else:
            warnings.warn("Code does not handle constraint beta(x)<=x yet; returnning 0")
            return 0
        new_coeffs = coeffs_bidder_shading + gradient_step_size * update_move.x
    return new_coeffs


def perform_multiple_gradient_ascent_steps(density_opponents, cdf_opponents,
                                                 knots_bidder,
                                                 coeffs_bidder_shading,
                                                 density_bidder, cdf_bidder,
                                                 x_0,
                                           assume_v_val_inc=False,
                                           lower_bound=0.0, upper_bound=1.0,
                                        gradient_step_size=10**-2,
                                        abs_max_size_move=1.0,
                                        number_steps=100,
                                        less_than_id_constraint=False,
                                           monotonicity_constraint=False,
                                           brute_force_comp_res_val=False,
                                           test_vals=[0.0,1.0],
                                           my_tol = 10**(-4)):
    """
    NOTE: this is really for one strategic; expanding on splines may not be such an intelligent idea
    as the optimal shading function may be discontinuous...
    :param density_opponents:
    :param cdf_opponents:
    :param knots_bidder:
    :param coeffs_bidder_shading:
    :param density_bidder:
    :param cdf_bidder:
    :param x_0:
    :param lower_bound:
    :param upper_bound:
    :param gradient_step_size:
    :param abs_max_size_move:
    :param number_steps:
    :param less_than_id_constraint:
    :return:
    """
    k=0
    current_coeffs = coeffs_bidder_shading
    current_v_value = compute_v_value_after_shading(knots_bidder, current_coeffs, density_bidder, cdf_bidder)
    beta_minus_one_current_x_0 = compute_x_0_from_v_value(current_v_value,test_vals,my_tol,assume_v_val_inc)
    bidder_shading = compute_bidder_shading(knots_bidder, coeffs_bidder_shading)

    if brute_force_comp_res_val is False:
        current_x_0=compute_reserve_value_from_0_of_v_value(beta_minus_one_current_x_0, bidder_shading, test_vals, my_tol=10 ** -4)
    else:
        current_x_0 = compute_x_0_from_v_value_brute_force(virtual_value_after_shading=current_v_value,
                                                       bidder_shading=bidder_shading,
                                                       cdf_opponents=cdf_opponents,
                                                       density_bidder=density_bidder, test_vals=[0.0, 1.0])[0]

    current_revenue = compute_current_revenue_bidder(cdf_opponents, bidder_shading, current_v_value,
                                                     density_bidder, current_x_0, upper_bound=1.0)

    # current_x_0 = x_0
    lst_reserve_values = []
    lst_reserve_prices = []
    lst_all_coeffs = []
    lst_current_revenue =[]

    lst_reserve_values.append(current_x_0)
    lst_all_coeffs.append(current_coeffs)
    lst_reserve_prices.append(bidder_shading(current_x_0))
    lst_current_revenue.append(current_revenue)


    while (k<number_steps):
        updated_vector=define_and_solve_one_gradient_step(density_opponents, cdf_opponents,
                                           knots_bidder,
                                           current_coeffs,
                                           density_bidder, cdf_bidder,
                                           current_x_0, assume_v_val_inc,
                                            lower_bound, upper_bound,
                                           gradient_step_size,
                                           abs_max_size_move,
                                           less_than_id_constraint,
                                            monotonicity_constraint)
        # print(updated_vector)
        #print(abs(max(current_coeffs - updated_vector)))
        if abs(max(current_coeffs-updated_vector))< my_tol :
            print('Not enough movement')
            print("Current index", k)
            print("Current reserve value", current_x_0)
            print("Current coeffs", current_coeffs)
            break
        current_coeffs = updated_vector

        current_v_value = compute_v_value_after_shading(knots_bidder, current_coeffs, density_bidder, cdf_bidder)
        beta_minus_one_current_x_0 = compute_x_0_from_v_value(current_v_value, test_vals, my_tol, assume_v_val_inc)
        bidder_shading = compute_bidder_shading(knots_bidder, current_coeffs)
        if brute_force_comp_res_val is False:
            current_x_0 = compute_reserve_value_from_0_of_v_value(beta_minus_one_current_x_0, bidder_shading, test_vals,
                                                                  my_tol=10 ** -4)
        else:
            current_x_0 = compute_x_0_from_v_value_brute_force(virtual_value_after_shading=current_v_value,
                                                               bidder_shading=bidder_shading,
                                                               cdf_opponents=cdf_opponents,
                                                               density_bidder=density_bidder, test_vals=[0.0, 1.0])[0]

        current_revenue = compute_current_revenue_bidder(cdf_opponents, bidder_shading, current_v_value,
                                                         density_bidder, current_x_0, upper_bound=1.0)
        #current_x_0 = compute_x_0_from_v_value(current_v_value,test_vals,my_tol,assume_v_val_inc)
        lst_reserve_values.append(current_x_0)
        lst_all_coeffs.append(current_coeffs)
        lst_reserve_prices.append(bidder_shading(current_x_0))
        lst_current_revenue.append(current_revenue)

        #print(current_coeffs)
        if (k % 25 == 0) :
            print("Number of steps", k)
            print("Current reserve value",current_x_0)
            print("Current coefficients", current_coeffs)
            print("Current revenue", current_revenue)
            print("knots",knots_bidder)
        k+=1

    return current_coeffs, lst_reserve_values, lst_all_coeffs, lst_reserve_prices, lst_current_revenue




def compute_performance_flat_then_truthful(cdf_opponents,
                                                transition_point,
                                                 density_bidder,
                                                cdf_bidder,
                                           test_vals=[0.0,1.0],
                                           my_tol = 10**(-4)
):
    """

    :param cdf_opponents:
    :param transition_point:
    :param density_bidder:
    :param cdf_bidder:
    :param test_vals:
    :param my_tol:
    :return:
    """
    knots_bidder = [transition_point]
    coeffs_bidder_shading = [transition_point,0.0,1]
    current_coeffs = coeffs_bidder_shading
    current_v_value = compute_v_value_after_shading(knots_bidder, current_coeffs, density_bidder, cdf_bidder)
    bidder_shading = compute_bidder_shading(knots_bidder, coeffs_bidder_shading)
    current_x_0 = compute_x_0_from_v_value_brute_force(virtual_value_after_shading=current_v_value,
                                         bidder_shading=bidder_shading,
                                         cdf_opponents=cdf_opponents,
                                         density_bidder=density_bidder,test_vals=[0.0, 1.0])


    #print("in the function", current_x_0)
    #print(transition_point)
    #print("reserve value", current_x_0)
    current_revenue = compute_current_revenue_bidder(cdf_opponents, bidder_shading, current_v_value,
                                                     density_bidder, current_x_0, upper_bound=1.0)


    lst_reserve_values = []
    lst_reserve_prices = []
    lst_all_coeffs = []
    lst_current_revenue =[]

    lst_reserve_values.append(current_x_0)
    lst_all_coeffs.append(current_coeffs)
    lst_reserve_prices.append(bidder_shading(current_x_0))
    lst_current_revenue.append(current_revenue)

    return current_coeffs, lst_reserve_values, lst_all_coeffs, lst_reserve_prices, lst_current_revenue
