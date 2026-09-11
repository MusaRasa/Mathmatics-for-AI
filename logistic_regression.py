# In logistic regression we have to state 1,0 
# Sigma = S
# The 1 probability formula ---> P(y=1|x,w) = S(wT*x)

# The 0 probability formula ---> P(y=0|x,w) = 1-S(wT*x)
# sigma formula ---> S = 1/1+e^-(wT*x)
import numpy as np
def logistic_regression(): # this is the  probability formula
    w = np.array([[2],[3],[5]],float)
    x = np.array([[4],[2],[3]],float)
    #Dot product formula wT*x
    print("W: ",w)
    print("X: ",x)
    wt = w.T
    print("W.T:",wt,"Shape w.t: ",wt.shape)
    z =wt@x
    print("w^T * x: ",z)
    # Sigma Formula S = 1/1+e^-dP
    s = 1/(1+np.exp(-z)) * 100 
    print("Sigmoid: ",s)
    p1 = s
    print("The probability of one: ",p1)
    P0 = 1-p1 * 100
    print("P0: ",P0)
    print("Shape W:",w.shape)
    print("shape x:",x.shape)
    print(w.shape," * ",x.shape)
    print("shape Dot Product: ",z.shape)
    if p1 >= 0.5:
        print("Positive_ ")
        print(f"The probability of One: {p1}  %")
    else:
        print("Negative_ ")
        print(f"The probability of zero: {P0}  %")
logistic_regression()

# print(pip list)