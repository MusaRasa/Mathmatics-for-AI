import torch as tr
from torch import nn
import matplotlib.pyplot as plt
"""
What is BCEWithlogits?
This loss combine a Sigmoid layer and the BCELoss in one single class.
Why we use? 
* it is faster than another function
* It's slove the overfittong and underfitting.
"""
loss_fn = nn.BCEWithLogitsLoss()
output = tr.tensor([0.1,-1.2,2.0,0.9]) # Logits
target = tr.tensor([0.,0.,1,1]) # Labels
loss = loss_fn(output,target)

plt.scatter(output,target)
plt.grid()
plt.show()
print(loss)
if loss >=0.5:
    print("Yes")
else:
    print("No")
