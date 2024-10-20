a=9-12/3+3*2-1
#a=?
a=9-4+3*2-1
a=9-4+6-1
a=5+6-1
a=11-1
a=10

a=2*3+4%5-3//2+6
a=6+4-1+6
a=10-1+6
a=15
print(a)

m=-43|8&0|-2
m=-43|0|-2
m=1|-2
m=1
print(m)

for i in range(1,11):
    print(f"2 x {i}={2 * i}")

a=[17,14,5,20,12]
temp=0
for i in range(0,len(a)-1):
    for j in range(i+1,len(a)):
        if a[i]>a[j]:
          temp=a[i]
          a[i]=a[j]
          a[j]=temp
print(a)

a=[1,2,3,4,5,6,7,8,9,10]
iteam=int(input("enter  no to find"))
first=0
last=10  
step=0
while(first<=last):
    mid=(last+first)//2
    if(iteam==a[mid]):
        flag=1
        step+=1
        break
    elif(iteam > a[mid]):
        first=mid+1
        step+=1
    else:
       last=mid-1
       step+=1
if(flag==1):
    print("number found")
    print("step",step)
    
else:
    print(404)

# def minor(m, n, matrix_name, minor_value):
#     minor_data = []
#     for i in range(0, m):
#         temp = []
#         for j in range(0, n):
#             if i != minor_value[0] and j != minor_value[1]:
#                 q = matrix_name[i][j]
#                 temp.append(q)
#         if temp != []:
#             minor_data.append(temp)
#     return numpy.array(minor_data)


# def cofactor(m, n, matrix_name):
#     cofactor_data = []
#     for i in range(0, m):
#         temp = []
#         for j in range(0, n):
#             minor_value = (i, j)
#             minor_matrix = minor(m, n, matrix_name, minor_value)
#             if m == 3:
#                 c = ((-1) ** (i + j)) * (minor_matrix[0][0] * minor_matrix[1][1] - minor_matrix[0][1] * minor_matrix[1][0])
#             elif m == 2:
#                 c = ((-1) ** (i + j)) * (minor_matrix[0][0])
#             temp.append(c)
#         cofactor_data.append(temp)
#     return numpy.array(cofactor_data)


# def det(m, n, cofactor_name, matrix_name):
#     i = 0
#     det_value = 0
#     for j in range(0, n):
#         det_value += cofactor_name[i][j] * matrix_name[i][j]
#     return det_value


# m1 = int(input("Enter Number of rows of the First Matrix = "))
# n1 = int(input("Enter Number of columns of the First Matrix = "))
# matrix1 = matrix(m1, n1)
# m2 = int(input("Enter Number of rows of the Second Matrix = "))
# n2 = int(input("Enter Number of columns of the Second Matrix = "))
# matrix2 = matrix(m2, n2)
# print(matrix1)
# print(matrix2)
# op = input("Enter operation (+, -, *, adj, det, inv) = ")
# if m1 == m2 and n1 == n2:
#     if op == "+":
#         print(matrix1 + matrix2)
#     elif op == "-":
#         print(matrix1 - matrix2)
# if n1 == m2 and op == "*":
#     e = []
#     r = []
#     cell_total = 0
#     for i in range(0, m1):
#         for j in range(0, n2):
#             for k in range(0, n1):
#                 q = matrix1[i][k] * matrix2[k][j]
#                 cell_total += q
#             e.append(cell_total)
#             cell_total = 0
#         r.append(e)
#         e = []
#     product = numpy.array(r)
#     print(f"Multiplication of Matrices is \n{product}")
# if op == "adj" or op == "inv" or op == "det" or op == "t":
#     which_one = input("Which matrix (First / Second)?").lower()
#     if which_one == "first":
#         m_, n_ = m1, n1
#         matrix_ = matrix1
#     elif which_one == "second":
#         m_, n_ = m2, n2
#         matrix_ = matrix2
#     cofactor_result = cofactor(m_, n_, matrix_)
#     adj = cofactor_result.transpose()
#     det_ = det(m_, n_, cofactor_result, matrix_)
#     if op == "adj":
#         print(adj)
#     elif op == "det":
#         print(det_)
#     elif op == "inv":
#        inv_ = adj/det_
#         print(inv_)
#     elif op == "t":
        # print(matrix_.transpose())

##

# import cmath #for handeling complex square roots

# def solve_quadratic(a,b,c):
#     #calculate the discriment
#     D=b**2 - 4*a*c

#     #calculate the two equations using the quadratic formula
#     root1=(-b + cmath.sqrt(D)) / (2*a)
#     root2=(-b - cmath.sqrt(D)) / (2*a)

#     return root1,root2
# def main():
#     print("solve the quadratic equation ax^2 + bx + c = 0")

#     #input coefficiants a,b and c
#     a=float(input("enter coefficient a: "))
#     b=float(input("enter coefficient b: "))
#     c=float(input("enter coefficient c: "))

#     if a==0:
#         print("coefficient 'a' cannot be 0 in a quadratic equation")
#         return
    
#     #solve the quadratic equation
#     roots=solve_quadratic(a,b,c)

#     #display the results
#     print(f"the roots are:{roots[0]} and {roots[1]}")

# if  __name__=="__main__":
#     main()

##


# def divcomplex(z1,z2):
#     return z1 / z2
# z1=complex(2,3)
# z2=complex(4,5)

# print(" division is ",divcomplex(z1,z2))

# def complex_division(numerator, denominator):
#     # Extract real and imaginary parts of numerator
#     a, b = numerator.real, numerator.imag
    
#     # Extract real and imaginary parts of denominator
#     c, d = denominator.real, denominator.imag
    
#     # Conjugate of the denominator
#     conjugate_denominator = complex(c, -d)
    
#     # Multiply numerator and denominator by the conjugate of the denominator
#     numerator_conjugate = numerator * conjugate_denominator
#     denominator_conjugate = denominator * conjugate_denominator
    
#     # Now denominator will be a real number (c^2 + d^2)
#     real_numerator = numerator_conjugate.real
#     imag_numerator = numerator_conjugate.imag
#     denominator_real = denominator_conjugate.real
    
#     # Resulting real and imaginary parts
#     real_part = real_numerator / denominator_real
#     imag_part = imag_numerator / denominator_real
    
#     return complex(real_part, imag_part)

# # Example usage
# numerator = complex(2, 3)
# denominator = complex(1, 2)
# result = complex_division(numerator, denominator)
# print(result)


# a=[1,2,3,4,5]
# a.append(6)
# print(a)
# a.insert(0,0)
# print(a)
# b=[7,8,9]
# a.extend(b)
# print(a)
# c=a.index(8)
# print(c)
# a.sort()
# print(a)
# a.reverse()
# print(a)


# a=list(input("enter the elements:  "))
# print("list before sorting is ",a)
# for i in range (len(a)):
#     for j in range(i+1,len(a)):
#         if a[i]<a[j]:
#             print(a)
#         elif a[i]>a[j]:
#             temp=a[i]
#             a[i]=a[j]
#             a[j]=temp
#             print("list after sorting is: ",a)

# a=[1,2,3,4,5]
# i=0
# sum=0
# while i<len(a):
#     sum=sum+a[i]
#     i=i+1
# print(sum)

# a=1
# while(a==1):
#     n=int(input("enter the number"))
#     print("you entered",n)

# num1=int(input("enter num1 : "))
# num2=int(input("enter num2 : "))
# def gcd(a,b):
#     while b:
#         a,b = b, a % b
#     return a
# print(f"the GCD of {num1} and {num2} is {gcd(num1, num2)}")

# num1=int(input("enter num1 : "))
# num2=int(input("enter num2 : "))
# def gcd(a, b):
#     gcd_value = 1
#     for i in range(1, min(a, b) + 1):
#         if a % i == 0 and b % i == 0:
#             gcd_value = i
#     return gcd_value
# print(f"the GCD of {num1} and {num2} is {gcd(num1, num2)}")

# a=int(input("enter a : "))
# b=int(input("enter b : "))
# for i in range (1,a+1):
#     if(a%i==0 and b%i==0):
#         gcd=i
# print(gcd)

# def find_gcd(x,y):
#     if (y==0):
#         return x
#     else:
#         return find_gcd(y,x%y)
# x=int(input("enter x: "))
# y=int(input("enter y: "))
# num= find_gcd(x,y)
# print("GCD of two number is: ")
# print(num)      


