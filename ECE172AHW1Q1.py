import numpy as np 

# part 1
A = np.array([[60, -15, 12, 22, 68], [-97, 28, 91, -49, 7], [9, 57, -91, 91, -88], [29, -92, 42, 38, 99], [-58, 91, 90, 95, -62]])
B = np.array([[1, 1, 0, 1,1], [1,0,1,1,0], [1,0,0,1,0], [1,1,0,0,1], [0,0,1,0,0]])

print("A =", A)

print("B=", B)
print(' ')

# part 2
print(A<-70)
rows_A, cols_A = np.where(A<-70)
print("rows of A<-70 = ", rows_A)
print("columns of A<-70 = ", cols_A)
print(' ')

# part 3
C = np.multiply(A,B)
print("C =", C)
print(' ')

# part 4
row_5_C = C[4,:]
col_3_C = C[:,2]
#inner_product_C = row_5_C @ col_3_C
inner_product_C = np.dot(col_3_C, row_5_C)
print("inner product = ", inner_product_C)
print(" ")

# part 5
max_col_4 = (C[:,3].max())
print("max at col 4 =", max_col_4)
row_C = np.where(C[:,3] == max_col_4)

print("row of max element= ", row_C)
print("col of max element=", 3)
print(" ")


# part 6
row_1_C = C[0]
print("1st row of C =", row_1_C)
D = np.multiply(row_1_C, C)
print("D =", D)
print(" ")

# part 7
row_5_D = C[4,:]
col_3_D = C[:,2]
#inner_product_D = row_5_D @ col_3_D
inner_product_D = np.dot(col_3_D, row_5_D)
print("inner product = ", inner_product_D)