import torch
import addition_extension
import numpy as np
import time
tensor1 = torch.randn(10000).to('cuda')
tensor2 = torch.randn(10000).to('cuda')



s=time.time()
result = addition_extension.vec_add(tensor1,tensor2)
print(time.time()-s)

print(np.allclose(result.cpu(), (tensor1+tensor2).cpu()))