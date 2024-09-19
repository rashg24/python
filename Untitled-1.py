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