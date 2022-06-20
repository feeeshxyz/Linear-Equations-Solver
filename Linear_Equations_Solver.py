
import numpy

def main(A, b):

    ### ROBUSTNESS CHECKS ###
    if b.shape[1] != 1:
        raise TypeError("Input b should be a vector") 
    if b.shape[0] != A.shape[0]:
        raise Exception("Input b should be the same length as A")
    if A.shape[0] != A.shape[1] or A.shape[0] == 1 or A.shape[0] == 0:
        raise Exception("Input A should be a *square* matrix of at least 2x2")
    if numpy.iscomplex(A.all()) or numpy.iscomplex(b.all()):
        raise Exception("Input A and B should only contain real numbers")

    n = A.shape[0] 
    
    # Create a vector of the max abs value from each row in A
    s = numpy.zeros((n,1))
    for i in range(n):
        s[i] = numpy.max(numpy.abs(A[i]))


    ### Scaled partial pivoting ###
    pivotRow = 1;
    pivotColumn = 1;
    
    for i in range(n):
        # Calculate ration vector for row i
        ratioVector = numpy.zeros((n,1))
        for j in range(i,n):
            ratioVector[j] = A[i,j] / s[j]

        # Swap the row with the highest scaled value with row i
        highIndex = numpy.where(ratioVector == numpy.max(numpy.abs(ratioVector)))[0][0]

        tmpRow = b[highIndex]
        b[highIndex] = b[i] # Swap b rows
        b[i] = tmpRow
        
        tmpRow = A[highIndex]
        A[highIndex] = A[i] # Swap A rows
        A[i] = tmpRow

        # Check that the pivot is not 0. 
        # If it is we find the next row with a non 0 in the same column as the pivot then add that row to the pivot row
        if A[pivotRow, pivotColumn] == 0: 
            for l in range(1,n):
                if A[l, pivotColumn] != 0:
                    A[pivotRow] = A[pivotRow] + A[l, pivotColumn]
                    b[pivotRow] = b[pivotRow] + A[l, pivotColumn]
                    # If the pivot is still 0 we can still use it
                    # just dont divide by 0.
                    if A[pivotRow, pivotColumn] != 0:
                        b[pivotRow] = b[pivotRow] / A[pivotRow, pivotColumn]
                        A[pivotRow] = A[pivotRow] / A[pivotRow, pivotColumn] # Pivot
        else:
            b[pivotRow] = b[pivotRow] / A[pivotRow, pivotColumn]
            A[pivotRow] = A[pivotRow] / A[pivotRow, pivotColumn] # Pivot = 1
        
        for j in range(pivotRow+1,n): # For each value in the column below the pivot
            b[j] = b[j] - b[pivotRow]*A[j,pivotColumn]
            A[j] = A[j] - A[pivotRow]*A[j,pivotColumn]

        pivotRow = i # Pivot should always b at i,i
        pivotColumn = i

        ### Calculate rank ###
        rank = n
        for i in range(1,n):
            if A[i].all() <= 1e-7:
                rank = rank - 1

        if rank != n:
            raise Exception("The matrix provided is rank defficient and so has no solution.")

        ### Solve Ux = b with back substitution ###
        x = numpy.zeros((n,1))

        # Answer vector is in the form x = [...,x,y,z]
        # Calculate z
        x[n] = b[n] / A[n,n]

        # Use z to calculate rest of the answer vector
        for i in range(n-1,1,-1):
            x[i] = numpy.dot(1/A[i,i], b[i]) - numpy.sum(numpy.dot(A[i,i+1:n], x[i+1:n]))

        return x

matrix = numpy.random.rand(5,5)*100
vector = numpy.random.rand(5,1)*100
main(matrix, vector)







