import torch
index = [1,0,2,1,0]
yhat = torch.tensor(index, dtype=int)
labels = [[1],[0],[1],[1],[1]]
label = torch.IntTensor(labels)
label = label.reshape(5)
from sklearn.metrics import precision_recall_fscore_support
p_class, r_class, f_class, support_micro = precision_recall_fscore_support(label, yhat, average='macro')
print(p_class, r_class, f_class, support_micro)
