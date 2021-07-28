import numpy as np        # for working with matricies and vectors
import warnings           # for error handling


def qr_mod_gram_schmidt(A_arg: np.matrix):
    """
        qr_mod_gram_schmidt(A: np.matrix) - solves finds QR Decomposition by the Modified Gram-Schmidt Algorithm

        Arguments:
            * A_arg - matrix

        Return:
            Tuple - two matrices Q - orthogonal and R - upper triangular
    """
    A = np.copy(A_arg)
    n = A.shape[0]
    R, Q = np.zeros(A.shape), np.zeros(A.shape)
    for k in range(n):
        s = 0
        for j in range(n):
            s += A[j, k]**2
            R[k, k] = np.sqrt(s)
        for j in range(n): Q[j, k] = A[j, k]/R[k, k]
        for i in range(k, n):
            s = 0
            for j in range(n):
                s += A[j, i] * Q[j, k]
                R[k, i] = s
            for j in range(n): A[j, i] = A[j, i] - R[k, i] * Q[j, k]
    return np.asmatrix(Q), np.asmatrix(R)


def qr_mod_algorithm(A: np.matrix, Kmax: int, delta: float) -> np.array:
    """
        qr_mod_algorithm(A: np.matrix, K_max: int, delta: float) - finds the eigenvalues of a matrix

        Arguments:
            * A - matrix
            * Kmax - maximum number of steps before stopping the iterative process
            * delta - value of stopping the iterative process by the criterion of proximity of neighboring approximations

        Return:
            np.array - matrix eigenvalues
    """
    if Kmax < 1:
        warnings.warn('Number of iterations must be a positive number')
        return
    Ak = np.copy(A)
    t = 0
    I = np.identity(A.shape[0])
    d = delta
    eigvals = []
    k = 0
    while k < Kmax and d >= delta:
        Q, R = qr_mod_gram_schmidt(Ak - t * I)
        Ak = np.matmul(R,Q) + t * I if k else np.matmul(R, Q)
        t = Ak[-1, -1]
        eigvals.append(np.diagonal(Ak))
        d = np.linalg.norm(eigvals[-1] - eigvals[-2]) if k else delta
        k += 1
    print(f'Number of iterations it took to find the solution: {k}\n')
    return eigvals[-1]



if __name__ == '__main__':
    A1 = np.matrix([[4.33, -1.12, -1.08, 1.14],
                   [-1.12, 4.33, 0.24, -1.22],
                   [-1.08, 0.24, 7.21, -3.22],
                   [1.14, -1.22, -3.22, 5.43]],
                   dtype=np.dtype(np.float64))

    A2 = np.matrix([[1.00, 0.42, 0.54, 0.66],
                   [0.42, 1.00, 0.32, 0.44],
                   [0.54, 0.32, 1.00, 0.22],
                   [0.66, 0.44, 0.22, 1.00]],
                   dtype=np.dtype(np.float64))

print(25*'-'+'TEST 1'+25*'-')
real_res1 = qr_mod_algorithm(A1, 200, 10**-20)
np_res1 = np.linalg.eigvals(A1)
print(f"Received answer\n{real_res1}\n\nNumPy's function answer\n{np_res1}")

print('\n\n'+25*'-'+'TEST 2'+25*'-')
real_res2 = qr_mod_algorithm(A2, 200, 10**-20)
np_res2 = np.linalg.eigvals(A2)
print(f"Received answer\n{real_res2}\n\nNumPy's function answer\n{np_res2}")
