import numpy as np        # for working with matricies and vectors
from typing import Union  # to work with typing
import warnings           # for error handling


def gaussian_elimination(A_arg: np.matrix, f_arg: Union[np.matrix, np.array]) -> Union[np.matrix, np.array]:
    """
        gaussian_elimination(A: np.matrix, f: Union[np.matrix, np.array]) - solves the matrix equation by the Gaussian method with Partial Pivot Selection

        Arguments:
            * A_arg - coefficient matrix
            * f_arg - right side of the system

        Return:
            Union[np.matrix, np.array] - answer to solving a system of linear algebraic equations
    """
    A, f = np.copy(A_arg), np.copy(f_arg)  # copy the arguments so as not to 'dirty' them
    for i in range(len(A)):
        column = np.abs(A[i:, i])      # take the `i`-th column by modulo
        leading_elem = np.max(column)  # by the Partial Pivot Selection we find the leading element
        if leading_elem == 0.:  # check the determinant (if pivot == 0 then `det(A)` = 0 => no solutions)
            warnings.warn("Determinant is 0")  # display an error
            return  # we end to execution of the program
        if np.where(column == leading_elem)[0][0] != 0:  # do we need to change lines (?)
            pos_max = column.argmax() + i      # find out the line number of the leading element
            A[[i, pos_max]] = A[[pos_max, i]]  # swap rows in matrix `A`
            f[[i, pos_max]] = f[[pos_max, i]]  # swap rows in matrix `f`
        for j in range(i+1, len(A)):   # making the upper triangular matrix
            coef = -(A[j, i]/A[i, i])  # count the coefficient
            A[j] = coef * A[i] + A[j]  # multiply the `i` row and add to row `j`
            f[j] = coef * f[i] + f[j]
    n = f.shape[0]  # dimension of our answer
    X = np.zeros(shape=f.shape)   # fill our future solution with zeros
    X[n-1] = f[n-1]/A[n-1, n-1]   # solve the last equation
    for i in range(n-2, -1, -1):  # calculates values starting from the end
        sum_elem = sum(A[i, j] * X[j] for j in range(i+1, n))  # for known `x` we sum up the coefficients
        X[i] = (f[i] - sum_elem)/A[i, i]  # find `x`
    return X  # return the answer to check the result


if __name__ == '__main__':
    A = np.matrix([[1.00, 0.17, -0.25, 0.54],
                   [0.47, 1.00, 0.67, -0.32],
                   [-0.11, 0.35, 1.00, -0.74],
                   [0.55, 0.43, 0.36, 1.00]],
                   dtype=np.dtype(np.float64))

    f = np.array([0.3, 0.5, 0.7, 0.9],
                 dtype=np.dtype(np.float64))

    X = gaussian_elimination(A, f)
    np.testing.assert_allclose(np.linalg.solve(A, f), X)

    print(X)
