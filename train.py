from gensim.models import word2vec as w2v
from preprocess import build_text_model
from dictionary import process_text
import torch
global BATCHSIZE

BATCHSIZE = 32

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

def encode(sentences,labels,word_to_index,total_word_count):
    samples,labels = process_text(sentences,labels,word_to_index, total_word_count)
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
    print("词语数量：",len(words))

    vectors = model.wv.vectors # 词向量集
    #print(model.wv.most_similar('医生')) #最相似的词
    #print(model.wv['医生'])

    word_to_index,index_to_vec = create_dictionary(model)
    # 注意：这里只用了word_to_index,没有调用预训练向量index_to_vec
    # 两个都是字典

    total_word_count = len(word_to_index)
    print("共计词数：",total_word_count)
    sentence2index = dict(enumerate(sentences_test))
    samples, labels = encode(sentences,labels,word_to_index,  total_word_count)
    samples_test,labels_test = encode(sentences_test,labels_test,word_to_index,total_word_count)

    import torch.utils.data as data
    batch_size = BATCHSIZE
    train_set = data.TensorDataset(samples,labels)
    train_iter = data.DataLoader(train_set, batch_size, shuffle=False,drop_last=False)

    test_set = data.TensorDataset(samples_test,labels_test)
    test_iter = data.DataLoader(test_set, batch_size, shuffle=False,drop_last=False)
    return train_iter,test_iter,vectors,total_word_count,sentence2index

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
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=256,hidden_size=hidden_size,batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_size*50,6))
    def forward(self,x):
        #print("x", x.size())
        x = self.embeds(x.to(device))
        #print("x_embded",)
        size = x.size(0)
        h0 = torch.randn(1, size, self.hidden_size)
        out,hidden = self.rnn(x.to(device),h0.to(device))
        #print("out",out.size())
        out = out.reshape(size,-1)
        #print("out_reshape", out.size())
        #print("h",hidden.size())
        return self.mlp(out)

class RNN(nn.Module):
    def __init__(self, input_size,hidden_size,total_word_count):
        super(RNN,self).__init__()
        self.embeds = nn.Embedding(total_word_count,embedding_dim=256)
        # batch BATCHSIZE length 50 embed 256 hidden 32
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=256,hidden_size=hidden_size,batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_size*50,6))
    def forward(self,x):
        #print("x", x.size())
        x = self.embeds(x.to(device))
        #print("x_embded",)
        size = x.size(0)
        h0 = torch.randn(1, size, self.hidden_size)
        out,hidden = self.rnn(x.to(device),h0.to(device))
        out = out.reshape(size,-1)
        return self.mlp(out)



class BiLSTM_Attention(nn.Module):
    def __init__(self, total_word_count, hidden_size, num_layers):
        super(BiLSTM_Attention, self).__init__()
        self.word_embeddings = nn.Embedding(total_word_count+2, 50)
        self.encoder = nn.LSTM(input_size=50,hidden_size=hidden_size, num_layers=num_layers,batch_first=True,bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(
            hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1)) #对hiddensize进行降维，以便得到每个h的注意力权重
        self.decoder = nn.Linear(2 * hidden_size, 6)

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        # inputs的形状是(batch_size,seq_len)
        embeddings = self.word_embeddings(inputs.to(device)).to(device)
        # 提取词特征，输出形状为(batch_size,seq_len,embedding_dim)
        # rnn.LSTM只返回最后一层的隐藏层在各时间步的隐藏状态。
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # outputs形状是(batch_size,seq_len, 2 * num_hiddens)
        #x = outputs.permute(1, 0, 2)
        # x形状是(batch_size, seq_len(timestep), 2 * num_hiddens)
        x = outputs.to(device)
        #print(x.size())
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega)).to(device)
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega).to(device)
        # att形状是(batch_size, seq_len, 1)
        import torch.nn.functional as F
        att_score = F.softmax(att, dim=1).to(device)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score.to(device)
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        feat = torch.sum(scored_x, dim=1).to(device)  # 加权求和
        # feat形状是(batch_size, 2 * num_hiddens)
        outs = self.decoder(feat)
        # out形状是(batch_size, 6)
        return outs.to(device)


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size,hidden_size,total_word_count,batch_size=BATCHSIZE):
        super(LSTM,self).__init__()
        self.embeds = nn.Embedding(total_word_count, embedding_dim=1024)
        self.gru = nn.LSTM(input_size=1024,hidden_size=hidden_size,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.mlp = nn.Sequential(nn.BatchNorm1d(2*hidden_size*50),nn.Linear(2*hidden_size*50,64),nn.Hardswish(),self.dropout,nn.Linear(64,6))
        self.hidden_size = hidden_size
    def forward(self,x):
        x = self.embeds(x.to(device))
        size = x.size(0)
        h0 =  torch.randn(2,size,self.hidden_size).to(device)
        c0 = torch.randn(2,size, self.hidden_size).to(device)
        out,hidden = self.gru(x,(h0,c0))
        out = out.reshape(size, -1)
        return self.mlp(out)

# 定义GRU模型
class GRU(nn.Module):
    def __init__(self, input_size,hidden_size,total_word_count,batch_size):
        super(GRU,self).__init__()
        self.embeds = nn.Embedding(total_word_count, embedding_dim=256)
        self.gru = nn.GRU(input_size=256,hidden_size=hidden_size,batch_first=True)
        self.mlp = nn.Linear(hidden_size*50,6)
        self.hidden_size = hidden_size
    def forward(self,x):
        x = self.embeds(x.to(device))
        #print(x.shape) # 30*50*256
        shape = x.size(0)
        h0 =  torch.randn(1,shape,self.hidden_size).to(device)
        out,hidden = self.gru(x.to(device),h0.to(device))
        shape = out.size(0)
        out = out.reshape(shape, -1)
        return self.mlp(out)


def calculate(net,dataset,loss):
    total_loss ,count = 0,0
    with torch.no_grad():
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

def evaluate(net,dataset,output=False):
    total_acc , count = 0,0
    yhat_list = torch.empty(1)
    labels_list = torch.tensor([0])
    encoder = {'neural': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'fear': 4, 'surprise': 5}
    decoder = dict(map(reversed, encoder.items()))
    with torch.no_grad():
        for instance, label in dataset:
            net.eval()
            prediction = net(instance)
            #print(prediction.size())
            m = nn.LogSoftmax(dim=1) #注意此处的维数
            y_out = m(prediction)
            value, index = y_out.max(axis=1)
            yhat = torch.tensor(index, dtype=int)
            size = label.size(0)
            label = label.reshape(size)
            yhat_list = torch.cat((yhat_list.long(),yhat.long()),0)
            labels_list = torch.cat((labels_list.long(),label.long()),0)
        #print(yhat_list.shape)
        yhat_list = del_tensor_ele(yhat_list, 0)
        #print(yhat_list.shape)
        labels_list = del_tensor_ele(labels_list, 0)
        #print(labels_list.shape)
        #if output:
            # for sentiment in labels_list.numpy().tolist():
            #     print(decoder[sentiment])
        import numpy as np
        from sklearn.metrics import precision_recall_fscore_support,classification_report
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(labels_list,yhat_list,average='macro')
        # print('Marco Precision:', p_class)
        # print('Marco Recall:', r_class)
        #print('Marco F1：', f_class)
        #print('Confusion Matrix:\n', classification_report(labels_list, yhat_list, labels=[0, 1, 2, 3, 4, 5]))
        hit = (yhat_list == labels_list).sum().item()
        total_acc += hit
        count = labels_list.shape[0]
        #print("sample size:",count)
        acc = total_acc/count
        report = classification_report(labels_list, yhat_list, labels=[0, 1, 2, 3, 4, 5])
        return acc,p_class,r_class,f_class,report,yhat_list,labels_list

def train(net,loss,optimizer,train_iter,test_iter,index2sentence):
    net = net.to(device)
    #acc_train,prec_tr,recall_tr,f1_tr,report_tr,yhat_list,label_list = evaluate(net, train_iter)
    #acc_test,prec_te,recall_te,f1_te,report_te,yhat_list,label_list = evaluate(net, test_iter)
    #print( "initial acc_train", acc_train, "initial acc_test", acc_test)
    num_epochs = 31
    score_log = []
    for epoch in range(num_epochs):
        for x, y in train_iter:
            #print(x.shape,y.shape)
            yhat = net(x)
            yhat = yhat.view(len(yhat), -1)
            l = loss(yhat, y.long().squeeze().to(device))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        #loss_train = calculate(net, train_iter, loss)
        loss_test = calculate(net, test_iter, loss)
        acc_train,prec_tr, recall_tr, f1_tr,report_tr,yhat_list,label_list  = evaluate(net, train_iter)
        acc_test,prec_te,recall_te,f1_te,report_te,yhat_list_te,label_list_te  = evaluate(net, test_iter,output=True)
        print("epoch",epoch,"*test*  ","f1:",f1_te,"loss:", loss_test, "precision:",prec_te,"recall:",recall_te,"acc(hand):",acc_test)
        print("train f1:",f1_tr)
        print(report_te)
        score_log.append([f1_tr, f1_te])
        if epoch!=0 and epoch% 10 == 0:
                import matplotlib
                matplotlib.use('Agg')
                from matplotlib import pyplot as plt
                plt.figure(figsize=(10, 5), dpi=300)
                import numpy as np
                score_log2 = np.array(score_log)
                print(score_log2)
                plt.plot(score_log2[:,0], linewidth=2,c='green',label="train set")
                plt.plot(score_log2[:,1], linewidth=2,c='red',label="test set")
                plt.legend()
                plt.title('F1-socre')
                plt.xlabel('epoch')
                import os
                print(os.getcwd())
                # save result
                plt.savefig(os.getcwd() + "/result/this.png")
                #plt.show()
        if epoch == 1000:
            print("epoch",epoch,report_te)
            wrong_items = []
            encoder = {'neural': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'fear': 4, 'surprise': 5}
            decoder = dict(map(reversed, encoder.items()))
            for i in range(len(yhat_list_te)):
                if yhat_list_te[i] != label_list_te[i]:
                    wrong_items.append(
                        [index2sentence[i], decoder[yhat_list_te[i].item()], decoder[label_list_te[i].item()]])
            for item in wrong_items:
                print(item[0], "prediction:", item[1], "label:", item[2])

    


loss = nn.CrossEntropyLoss()
train_iter, test_iter,vector,total_word_count,index2sentence = get_tensor_set()
#rnn = RNN(input_size=50,hidden_size=32,total_word_count = total_word_count)
#rnn = LSTM(input_size=50,hidden_size=256,total_word_count = total_word_count,batch_size=BATCHSIZE)
rnn = BiLSTM_Attention(total_word_count = total_word_count,hidden_size=256,num_layers=1).to(device)
#rnn = GRU(input_size=50,hidden_size=32,total_word_count = total_word_count,batch_size=BATCHSIZE)
from test import *
#rnn = Transformer(total_word_count)

print(rnn)
#optimizer = torch.optim.Adam(rnn.parameters(),lr=2*1e-4,weight_decay=1*1e-3)
optimizer = torch.optim.Adamax(rnn.parameters(),lr=1e-4,weight_decay=0.8*1e-4) # 0.4791
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 定义衰减策略

import warnings
warnings.filterwarnings("ignore")
# 传入模型：rnn
train(rnn,loss,optimizer,train_iter,test_iter,index2sentence)



