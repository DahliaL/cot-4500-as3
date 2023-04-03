from decimal import Decimal
import numpy as np

print(np.__file__)

def f(t, y):
    return t - y**2


def problem_1():
    # problem 1
    t0, tn = 0, 2
    h = (tn - t0) / 10

    y0 = 1

    t = [t0]
    y = [y0]

    for i in range(10): # this does what the modified_eulers.py does
        y_next = y[-1] + h * f(t[-1], y[-1])
        t_next = t[-1] + h
        t.append(t_next)
        y.append(y_next)

    print("{:.16f}\n".format(y[10]))


def problem_2(start, end, iter, beg_val):
    steps = (end - start) / iter
    x, y = beg_val

    for i in range(iter):
        k1 = steps * f(x, y)
        k2 = steps * f(x + steps/2, y + k1/2)
        k3 = steps * f(x + steps/2, y + k2/2)
        k4 = steps * f(x + steps, y + k3)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += steps
    
    print("{:.16f}\n".format(y))


def problem_3(arr):
    arr = arr.astype(float)
    n = arr.shape[0]
    
    for i in range(n):
        for j in range(i+1, n):
            pivot = arr[j, i] / arr[i, i]
            arr[j,:] = arr[j,:] - arr[i, :] * pivot

    # backward substitution
    col = len(arr[1]) - 1
    b1 = np.ones(col)
    b2 = np.zeros(col)
    j = col - 1
    
    for i in range(n-1, -1, -1):
        b2 = b1 * arr[i, :col]
        
        if j+1 == col:
            b1[j] = arr[i, col]/b2[j]
        else:
            b1[j] = (arr[i, col] - sum(b2[j+1:col])) / b2[j]
        
        j = j - 1
   
    print(b1, end = '\n\n')

def problem_4(mat):
    # Get the shape of the matrix
    n = mat.shape[0]

    # Initialize L and U matrices with zeros
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
    # U matrix
        for i in range(j, n):
            U[j, i] = mat[j, i] - np.sum(L[j, :j] * U[:j, i])
        # L matrix
        for i in range(j+1, n):
            L[i, j] = (mat[i, j] - np.sum(L[i, :j] * U[:j, j])) / U[j, j]

# Set the diagonal elements of L to 1
    for i in range(n):
        L[i, i] = 1

    det_U = np.linalg.det(U)

    print("{:.16f}\n".format(det_U))

    print(L, end='\n\n')
    print(U, end = '\n\n')

def diag_dom(matrix):
    diagonal = np.abs(matrix.diagonal())
    others = np.sum(np.abs(matrix), axis=1) - diagonal
    
    return np.all(diagonal >= others)


def problem_5(array):
    print(diag_dom(array), end = '\n\n')


def pos_def(arr):
    # is it a square?
    if arr.shape[0] != arr.shape[1]:
        return False
    # is it symmetric
    if not np.allclose(arr, arr.T):
        return False
    # are all the eigenvalues positive
    eigenvalues = np.linalg.eigvals(arr)
    
    return np.all(eigenvalues > 0)


def problem_6(arr):
    print(pos_def(arr), end = '\n')
    

def main():
    problem_1()

    initpoint = (0, 1)
    problem_2(0, 2, 10, initpoint)
    
    problem_3(np.array(np.mat('2 -1 1 6; 1 3 1 0; 1 5 4 -3')))

    given_matrix = np.array([[9, 0, 5, 2, 1],
                            [3, 9, 1, 2, 1],
                            [0, 1, 7, 2, 3],
                            [4, 2, 3, 12, 2],
                            [3, 2, 4, 0, 8]])
    
    gm = np.array(np.mat('1 1 0 3; 2 1 -1 1; 3 -1 -1 2; -1 2 3 -1'), subok=True)

    p6 = np.array(np.mat('2 2 1; 2 3 0; 1 0 2'))

    problem_4(gm)

    problem_5(given_matrix)

    problem_6(p6)


if __name__ == "__main__":
    main()
