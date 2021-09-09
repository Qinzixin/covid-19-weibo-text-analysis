from gensim.models import word2vec as w2v
from preprocess import build_text_model
from dictionary import process_text
import torch

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
    batch_size = 64
    train_set = data.TensorDataset(samples,labels)
    train_iter = data.DataLoader(train_set, batch_size, shuffle=True)

    test_set = data.TensorDataset(samples_test,labels_test)
    test_iter = data.DataLoader(test_set, batch_size, shuffle=True)
    return train_iter,test_iter,vectors,total_word_count

global device

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0'
else:
    device = torch.device("cpu")

# 定义RNN模型
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, input_size,hidden_size,total_word_count):
        super(RNN,self).__init__()
        self.embeds = nn.Embedding(total_word_count,embedding_dim=256)
        self.h0 = torch.randn(1, 50, hidden_size)
        self.rnn = nn.RNN(input_size=256,hidden_size=hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size,6))
    def forward(self,x):
        x = self.embeds(x.to(device))
        #print(x.size())
        out,hidden = self.rnn(x.to(device),self.h0.to(device))
        return self.mlp(out)

def calculate(net,dataset,loss):
    total_loss ,count = 0,0
    for instace, label in dataset:
        input = instace
        yhat = net(input)
        yhat = yhat.view(len(yhat), -1)
        l = loss(yhat, label.long().squeeze())
        total_loss += l
        count += 1
    return (total_loss/count).data.item()

def evaluate(net,dataset):
    total_acc , count = 0,0
    for instance, label in dataset:
        prediction = net(instance)
        m = nn.LogSoftmax(dim=2) #注意此处的维数
        y_out = m(prediction)
        value, index = y_out.max(axis=2)
        yhat = torch.tensor(index, dtype=int)
        hit = (yhat == label).sum().item()
        total_acc += hit
        count += label.shape[0]
    return total_acc/count

def train(net,loss,optimizer,train_iter,test_iter):
    acc_train = evaluate(net, train_iter)
    acc_test = evaluate(net, test_iter)
    print( "initial acc_train", acc_train, "initial acc_test", acc_test)
    net = net.to(device)
    num_epochs = 1000
    for epoch in range(num_epochs):
        for x, y in train_iter:
            # print(x.shape,y.shape)
            yhat = net(x)
            yhat = yhat.view(len(yhat), -1)
            l = loss(yhat, y.long().squeeze())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        loss_train = calculate(net, train_iter, loss)
        loss_test = calculate(net, test_iter, loss)
        acc_train = evaluate(net, train_iter)
        acc_test = evaluate(net, test_iter)
        print("loss_train:",loss_train,"loss_test:",loss_test,"acc_train:",acc_train,"acc_test:",acc_test)


loss = nn.CrossEntropyLoss()
train_iter, test_iter,vector,total_word_count = get_tensor_set()
rnn = RNN(input_size=50,hidden_size=32,total_word_count = total_word_count)
optimizer = torch.optim.Adam(rnn.parameters(),lr=1e-4)
train(rnn,loss,optimizer,train_iter,test_iter)



