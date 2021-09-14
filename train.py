from gensim.models import word2vec as w2v
from preprocess import build_text_model
from dictionary import process_text
import torch
global BATCHSIZE

BATCHSIZE = 64

def create_dictionary(model):
    # 创建词典：只有词
    from gensim.corpora import Dictionary
    dictionary = Dictionary()
    dictionary.doc2bow(model.wv.index_to_key, allow_update=True)
    # 构建词到id的映射
    word_to_index = {word:index for index,word in dictionary.items()}
    # 构建从id到词向量的映射
    index_to_vec = {word_to_index.get(word): model.wv.__getitem__(word) for word in word_to_index.keys()}
    return word_to_index,index_to_vec

def encode(sentences,labels,word_to_index):
    samples,labels = process_text(sentences,labels,word_to_index)
    return samples,labels

def get_tensor_set():
    # 对**训练集和测试集融合**进行分词，生成词向量
    sentence_cut_name = 'splite_word_all.txt'
    model_file_name = 'model.dat'
    sentences,sentences_test,labels,labels_test = build_text_model(sentence_cut_name,model_file_name)
    tags = labels

    # 加载文本模型
    model = w2v.Word2Vec.load(model_file_name)
    words = model.wv.index_to_key # 词集

    vectors = model.wv.vectors # 词向量集
    #print(model.wv.most_similar('医生')) #最相似的词
    #print(model.wv['医生'])

    word_to_index,index_to_vec = create_dictionary(model)
    # 注意：这里只用了word_to_index,没有调用预训练向量index_to_vec
    # 两个都是字典

    total_word_count = len(word_to_index)
    print("共计词数：",total_word_count)
    samples, labels = encode(sentences,labels,word_to_index)
    samples_test,labels_test = encode(sentences_test,labels_test,word_to_index)

    import torch.utils.data as data
    batch_size = BATCHSIZE
    train_set = data.TensorDataset(samples,labels)
    train_iter = data.DataLoader(train_set, batch_size, shuffle=True,drop_last=True)

    test_set = data.TensorDataset(samples_test,labels_test)
    test_iter = data.DataLoader(test_set, batch_size, shuffle=True,drop_last=True)
    return train_iter,test_iter,vectors,total_word_count

global device

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    print("Working on GPU")
else:
    device = torch.device("cpu")

# 定义RNN模型
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, input_size,hidden_size,total_word_count):
        super(RNN,self).__init__()
        self.embeds = nn.Embedding(total_word_count,embedding_dim=256)
        # batch BATCHSIZE length 50 embed 256 hidden 32
        self.h0 = torch.randn(1, BATCHSIZE, hidden_size)
        self.rnn = nn.RNN(input_size=256,hidden_size=hidden_size,batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_size*50,6))
    def forward(self,x):
        #print("x", x.size())
        x = self.embeds(x.to(device))
        #print("x_embded",x.size())
        out,hidden = self.rnn(x.to(device),self.h0.to(device))
        #print("out",out.size())
        out = out.reshape(BATCHSIZE,-1)
        #print("out_reshape", out.size())
        #print("h",hidden.size())
        return self.mlp(out)

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size,hidden_size,total_word_count,batch_size=BATCHSIZE):
        super(LSTM,self).__init__()
        self.embeds = nn.Embedding(total_word_count, embedding_dim=256)
        self.gru = nn.LSTM(input_size=256,hidden_size=hidden_size,batch_first=True)
        self.h0 =  torch.randn(1,batch_size,hidden_size)
        self.c0 = torch.randn(1,batch_size, hidden_size)
        self.mlp = nn.Linear(hidden_size*50,6)
    def forward(self,x):
        x = self.embeds(x.to(device))
        out,hidden = self.gru(x,(self.h0,self.c0))
        out = out.reshape(BATCHSIZE, -1)
        return self.mlp(out)

# 定义GRU模型
class GRU(nn.Module):
    def __init__(self, input_size,hidden_size,total_word_count,batch_size=BATCHSIZE):
        super(GRU,self).__init__()
        self.embeds = nn.Embedding(total_word_count, embedding_dim=256)
        self.gru = nn.GRU(input_size=256,hidden_size=hidden_size,batch_first=True)
        self.h0 =  torch.randn(1,batch_size,hidden_size)
        self.mlp = nn.Linear(hidden_size*50,6)
    def forward(self,x):
        x = self.embeds(x.to(device))
        out,hidden = self.gru(x.to(device),self.h0.to(device))
        out = out.reshape(BATCHSIZE, -1)
        return self.mlp(out)

def calculate(net,dataset,loss):
    total_loss ,count = 0,0
    for instace, label in dataset:
        input = instace
        yhat = net(input)
        yhat = yhat.view(len(yhat), -1)
        l = loss(yhat, label.long().squeeze().to(device))
        total_loss += l
        count += 1
    return (total_loss/count).data.item()


def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def evaluate(net,dataset):
    total_acc , count = 0,0
    yhat_list = torch.empty(1)
    labels_list = torch.empty(1)
    for instance, label in dataset:
        prediction = net(instance)
        #print(prediction.size())
        m = nn.LogSoftmax(dim=1) #注意此处的维数
        y_out = m(prediction)
        value, index = y_out.max(axis=1)
        yhat = torch.tensor(index, dtype=int)
        label = label.reshape(BATCHSIZE)
        yhat_list = torch.cat((yhat_list,yhat),0)
        labels_list = torch.cat((labels_list,label),0)
    print(yhat_list.shape)
    yhat_list = del_tensor_ele(yhat_list, 0)
    print(yhat_list.shape)
    labels_list = del_tensor_ele(labels_list, 0)
    print(labels_list.shape)
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support,classification_report
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(labels_list,yhat_list,average='macro')
    # print('Marco Precision:', p_class)
    # print('Marco Recall:', r_class)
    print('Marco F1：', f_class)
    print('Confusion Matrix:\n', classification_report(labels_list, yhat_list, labels=[0, 1, 2, 3, 4, 5]))
    hit = (yhat_list == labels_list).sum().item()
    total_acc += hit
    count = labels_list.shape[0]
    print("sample size:",count)
    acc = total_acc/count
    return acc,p_class,r_class,f_class

def train(net,loss,optimizer,train_iter,test_iter):
    net = net.to(device)
    acc_train,prec_tr,recall_tr,f1_tr = evaluate(net, train_iter)
    acc_test,prec_te,recall_te,f1_te = evaluate(net, test_iter)
    print( "initial acc_train", acc_train, "initial acc_test", acc_test)
    num_epochs = 500
    score_log = []
    for epoch in range(num_epochs):
        for x, y in train_iter:
            # print(x.shape,y.shape)
            yhat = net(x)
            yhat = yhat.view(len(yhat), -1)
            l = loss(yhat, y.long().squeeze().to(device))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        loss_train = calculate(net, train_iter, loss)
        loss_test = calculate(net, test_iter, loss)
        acc_train,prec_tr, recall_tr, f1_tr = evaluate(net, train_iter)
        acc_test,prec_te,recall_te,f1_te = evaluate(net, test_iter)
        print("epoch",epoch,"*train*","f1:",f1_tr,"loss:",loss_train,"precision:",prec_tr,"recall:",recall_tr,"acc(hand):",acc_train)
        print("       ","*test*  ","f1:",f1_te,"loss:", loss_test, "precision:",prec_te,"recall:",recall_te,"acc(hand):",acc_test)
        score_log.append([f1_tr,f1_te])

    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 5), dpi=300)
    import numpy as np
    score_log = np.array(score_log)
    plt.plot(score_log[:0], linewidth=1,c='#6b016d')
    plt.plot(score_log[:, 1], c='#e765eb')
    plt.title('F1-socre')
    plt.xlabel('epoch')
    import os
    print(os.getcwd())
    plt.savefig(os.getcwd() + "/result/gru-scratch-loss.png")
    plt.show()

loss = nn.CrossEntropyLoss()
train_iter, test_iter,vector,total_word_count = get_tensor_set()
rnn = GRU(input_size=50,hidden_size=32,total_word_count = total_word_count,batch_size=BATCHSIZE)
optimizer = torch.optim.Adam(rnn.parameters(),lr=1e-4)
import warnings
warnings.filterwarnings("ignore")
# 传入模型：rnn
train(rnn,loss,optimizer,train_iter,test_iter)



