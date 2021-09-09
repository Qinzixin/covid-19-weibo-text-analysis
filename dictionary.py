from gensim.models import word2vec as w2v
from preprocess import build_text_model
import torch

# 将文本token为id列表
def get_tokenized_text(text,labels,word_to_index):
    if len(text) == len(labels):
        vol = len(text)
        for i in range(vol):
            temp = []
            # 对每一行文本进行分词
            import jieba
            textline = text[i]
            word_list = " ".join(jieba.cut(textline))
            for word in word_list:
                if (word in word_to_index.keys()):
                    temp.append(int(word_to_index[word]))
                else:
                    temp.append(0)
            yield [temp, labels[i]]

def padding(x,max_length):
    if len(x)>max_length:
        text = x[:max_length]
    else:
        text = x + [1] * (max_length - len(x))
    return text

# 文本处理为相同长度的序列
# 核心模块：文本转向量，向量转固定长度的张量
def process_text(text,labels,word_to_index):
    data = get_tokenized_text(text,labels,word_to_index)
    max_length = 50
    labeltensor = torch.IntTensor(labels)
    samples = []
    for content in data:
        text_to_sequence = padding(content[0],max_length)
        samples.append([text_to_sequence])
        print(text_to_sequence)
    sampletensor = torch.FloatTensor(samples) # Long type will cause error in training?
    #print(sampletensor)
    return sampletensor,labeltensor

