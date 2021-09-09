import json

#读取文件
def resolveJson(path):
    file = open(path, "rb")
    file_json = json.load(file)
    record = []
    for i in range(len(file_json)):
        fileJson =  file_json[i]
        id = fileJson["id"]
        content = fileJson["content"]
        label = fileJson["label"]
        record.append([id,content,label])
    return record

def build_text_model(sentence_cut_name,model_file_name):
    import os
    # 读取训练集数据
    path = os.getcwd() + r"/data/virus_train.txt"
    result = resolveJson(path)
    # 读取测试集数据
    path_test = os.getcwd() + r"/data/virus_eval_labeled.txt"
    result_test = resolveJson(path)

    # 提取句子
    sentences_tr = [i[1] for i in result]
    sentences_te = [i[1] for i in result_test]
    sentences = sentences_tr + sentences_te

    labels_tr = [i[2] for i in result]
    labels_te = [i[2] for i in result_test]

    mytext = "".join(str(x) for x in sentences)

    # 对每一行的标签进行one-hot
    def encode_label(labels):
        from numpy import asarray
        from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
        data = [[label] for label in labels]
        # define ordinal encoding
        encoder = OrdinalEncoder()
        # transform data
        labels = encoder.fit_transform(data)
        return labels

    labels_tr = encode_label(labels_tr)
    labels_te = encode_label(labels_te)

    # 中文分词
    import jieba
    mywords = " ".join(jieba.cut(mytext))

    # 生成词云，展示预处理结果
    text = " ".join(mywords)
    from wordcloud import WordCloud
    import os
    import PIL.Image as image
    import numpy as np
    print("Generating word cloud")
    wordcloud = WordCloud(
        font_path=os.getcwd() + "/font/kaiti.ttf",
        mask = np.array(image.open(os.getcwd() + "/font/map.jpg")),
        mode='RGBA',
        background_color='white',
        scale=1
    ).generate(text)
    image_produce = wordcloud.to_image()
    link = os.getcwd() + "/font/wordcloud.png"
    wordcloud.to_file(link)
    #image_produce.show()
    print("word cloud has been genrated in "+str(link))
    # 保存分词
    f = open('splite_word_all.txt', 'w', encoding='utf-8')
    f.write(mywords)
    f.close()

    # 计算词向量
    from gensim.models import word2vec as w2v

    # 模型训练，生成词向量
    import os
    sentences_cut = w2v.LineSentence(os.getcwd()+"/"+sentence_cut_name)
    model = w2v.Word2Vec(sentences_cut, vector_size=30, window=15, min_count=5,
                         workers=4)  # 参数含义：数据源，生成词向量长度，时间窗大小，最小词频数，线程数
    model.save(model_file_name)

    return sentences_tr,sentences_te,labels_tr,labels_te






