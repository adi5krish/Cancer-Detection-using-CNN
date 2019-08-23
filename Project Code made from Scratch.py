import numpy as np

def nonlin(x, deriv=False):
    if(deriv == True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))


matrix = np.loadtxt('data.csv',dtype='string',delimiter=',')


a = np.zeros(shape=(569,30))
val = np.zeros(shape=(569,1))

x = 0



for i in range(569):
    get = 0
    for j in range(32):
        if j!=1 and j!=0:
            a[i,get] = float(matrix[i,j])
            get = get+1
        if j==1 and matrix[i,j] == 'M' and x<30:
            val[x,0] = 0
            x = x+1
        if j==1 and matrix[i,j] == 'B' and x<30:
            val[x,0] = 1
            x = x+1



print(a)
print(val)


np.random.seed(1)

syn0 = 2*np.random.random((30,569)) - 1

syn1 = 2*np.random.random((569,1)) - 1



for j in xrange(1000):

    l0 = a
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    #backpropagation
    print("**")

    l2_error = val - l2
    #if j%10000 == 0:
        #print 'Error:' + str(np.mean(np.abs(l2_error)))

    l2_delta = l2_error * nonlin(l2, deriv = True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv = True)

    syn1 += l1.T.dot(l2_delta) * 0.01
    syn0 += l0.T.dot(l1_delta) * 0.01



def result(inputs):
    output_1 = nonlin(np.dot(inputs,syn0))
    output_2 = nonlin(np.dot(output_1,syn1))
    return output_2



print('output after training')
print(result(np.array([12.05,14.63,78.04,449.3,0.1031,0.09092,0.06592,0.02749,0.1675,0.06043,0.2636,0.7294,1.848,19.87,0.005488,0.01427,0.02322,0.00566,0.01428,0.002422,13.76,20.7,89.88,582.6,0.1494,0.2156,0.305,0.06548,0.2747,0.08301])))
