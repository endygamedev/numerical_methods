import numpy as np        # for working with matricies and vectors
import warnings           # for error handling


def is_positive_definite(A: np.matrix) -> bool:
    """
        is_positive_definite(A: np.matrix) - checks if a matrix is positive definite

        Arguments:
            * A: np.matrix - matrix

        Return:
            bool - result of checking
    """
    return np.all(np.linalg.eigvals(A) > 0)  # count the eigenvalues and check that they are all greater than 0


def is_symmetric(A: np.matrix) -> bool:
    """
        is_symmetric(A: np.matrix) - checks if a matrix is symmetric

        Arguments:
            * A: np.matrix - matrix

        Return:
            bool - result of checking
    """
    return np.allclose(A, A.T)  # compare the matrix and the transposed one
                                # if they 'equal' then the matrix is symmetric


def steepest_descent_method(A_arg: np.matrix, f_arg: np.array, K_max: int) -> np.array:
    """
        steepest_descent_method(...) - solves the SLAE by the steepest descent method

        Arguments:
            * A_arg - coefficient matrix
            * f_arg - right side of the system
            * K_max - criterion for terminating the iterative process by proximity to the solution

        Return:
            np.array - answer to solving a system of linear algebraic equations
    """
    A, f = np.copy(A_arg), np.copy(f_arg)  # copy the arguments so as not to 'dirty' them
    if not is_positive_definite(A):  # check if the matrix `A` is positive definite
        warnings.warn("Matrix is not positive definite")  # display an error
        return
    elif not is_symmetric(A):  # check whether the matrix `A` is symmetric
        warnings.warn("Matrix is not symmetric")  # display an error
        return
    elif K_max < 0:  # check `K_max`
        warnings.warn("The number of iterations cannot be negative")  # display an error
        return
    x = np.zeros(f.shape, dtype=np.dtype(np.float64))  # initial approximation
    for k in range(K_max):  # iterate to `K_max`
        r = np.squeeze(np.asarray(f - np.matmul(A, x)))  # find the residual vector
        alpha = (np.dot(r, r)/np.dot(np.matmul(A, r), r)).item(0)  # find `alpha`
        x = x + alpha*r  # find the `k`-th approximate solution
    return x  # return the answer to check the result


if __name__ == '__main__':
    A = np.matrix([[4.33, -1.12, -1.08, 1.14],
                   [-1.12, 4.33, 0.24, -1.22],
                   [-1.08, 0.24, 7.21, -3.22],
                   [1.14, -1.22, -3.22, 5.43]],
                   dtype=np.dtype(np.float64))

    f = np.array([0.3, 0.5, 0.7, 0.9],
                 dtype=np.dtype(np.float64))

    x = steepest_descent_method(A, f, 100)
    np.testing.assert_allclose(np.linalg.solve(A, f), x)

    print(x)
