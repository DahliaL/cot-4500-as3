from decimal import Decimal
import numpy as np

def f(t, y):
    return t - y**2


def problem_1(start, end, iterr, beg_val):
    h = (end - start) / iterr

    x0, y0 = beg_val

    t = [start]
    y = [y0]

    for i in range(10):
        y_next = y[-1] + h * f(t[-1], y[-1])
        t_next = t[-1] + h
        t.append(t_next)
        y.append(y_next)

    print("{:.5f}\n".format(y[10]))


def problem_2(start, end, iterr, beg_val):
    steps = (end - start) / iterr
    x, y = beg_val

    for i in range(iterr):
        s1 = steps * f(x, y)
        s2 = steps * f(x + steps/2, y + s1/2)
        s3 = steps * f(x + steps/2, y + s2/2)
        s4 = steps * f(x + steps, y + s3)
        
        y += (s1 + 2*s2 + 2*s3 + s4) / 6
        x += steps
    
    print("{:.5f}\n".format(y))


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

    print("{:.5f}\n".format(det_U))

    print(L, end='\n\n')
    print(U, end = '\n\n')

def diag_dom(matrix):
    diagonal = np.abs(matrix.diagonal()) #absolute val of diag elements
    others = np.sum(np.abs(matrix), axis=1) - diagonal # sum of rows minus diag elements
    
    return np.all(diagonal >= others)


def problem_5(array):
    print(diag_dom(array), end = '\n\n')


def pos_def(arr):
    # is it a square?
    if arr.shape[0] != arr.shape[1]:
        return False # it cant be positive definite
    # is it symmetric
    if not np.allclose(arr, arr.T):
        return False # it cant be either
    # are all the eigenvalues positive
    eigenvalues = np.linalg.eigvals(arr)
    
    return np.all(eigenvalues > 0)


def problem_6(arr):
    print(pos_def(arr), end = '\n')
    

def main():
    initpoint = (0, 1)
    problem_1(0, 2, 10, initpoint)

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
