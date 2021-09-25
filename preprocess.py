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


def clean(json_text):
    print("Clean data")
    # 加载停用词列表
    with open('cn_stopwords.txt', encoding='utf-8') as f_stop:
        stopwords = [line.strip() for line in f_stop]
        f_stop.close()

    for json_item in json_text:
        text = json_item[1]
        import re
        text = re.sub(r'\/\/\@.*?(\：|\:)', "",text) #清除被转发用户用户名
        text = re.sub(r'\#.*?\#',"",text) # 清除话题
        text = re.sub(r'\【.*?\】', "", text)  # 清除话题
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', "", text, flags=re.MULTILINE) # 清除连接
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]", "", text) # 去除中文标点符号
        # 去除停用词
        outstr=''
        for word in text.split():
            if word not in stopwords:
                if word != '/t':
                    outstr += word
        json_item[1] = outstr
    print("Data is cleaned!")
    return json_text

def text_argue(path):
    from backtrans import back_trans,back_trans_enhance
    file = open(path, "rb")
    file_json = json.load(file)
    cardinality = len(file_json)
    text_argued = []
    aug_num = 0
    for i in range(cardinality):
        #print("增广进度：",i/cardinality,'\r',flush=True)
        fileJson =  file_json[i]
        id = fileJson["id"]
        content = fileJson["content"]
        label = fileJson["label"]
        #读取之后是一个字典
        text = {"id": id, "content": content, "label": label}
        text_argued.append(text)
        sparse_set = ["sad","fear","surprise"]
        if label in sparse_set: # data argumentation
            content_trans = back_trans(content)
            new_text = {"id": cardinality + aug_num, "content": content_trans, "label": label}
            text_argued.append(new_text)
            aug_num = aug_num + 1
        '''
        if label == "surprise": # too sparse
            content_trans = back_trans_enhance(content)
            for i in range(3):
                new_text = {"id": cardinality + aug_num, "content": content_trans[i], "label": label}
                text_argued.append(new_text)
                aug_num = aug_num + 1'''
        if i % 100 == 0:
            import os
            file = os.getcwd() +r"/data/virus_train_arg2.txt"
            print("增广后的训练集长度：",len(text_argued))
            with open(file, 'w',encoding='utf-8') as f:
                json.dump(text_argued, f,ensure_ascii=False)
                print("制作进度:",i/cardinality)

def build_text_model(sentence_cut_name,model_file_name):
    import os
    # path = os.getcwd() + r"/data/virus_train.txt"
    # # 对训练集进行数据增广
    # text_argue(path)
    print("数据增广完毕")
    argumented_path = os.getcwd()+r"/data/virus_train_arg.txt"
    # 读取训练集数据
    result = resolveJson(argumented_path)
    # 读取测试集数据
    path_test = os.getcwd() + r"/data/virus_eval_labeled.txt"
    result_test = resolveJson(path_test)

    # 在这部分进行清洗
    result = clean(result)
    result_test = clean(result_test)

    # 提取句子
    sentences_tr = [i[1] for i in result]
    sentences_te = [i[1] for i in result_test]
    sentences = sentences_tr + sentences_te

    labels_tr = [i[2] for i in result]
    labels_te = [i[2] for i in result_test]

    mytext = "".join(str(x) for x in sentences)

    # 对每一行的标签进行one-hot
    def encode_label(labels,title):
        encoder = {'neural':0,'happy':1,'angry':2,'sad':3,'fear':4,'surprise':5}
        labels2 = [encoder[label] for label in labels]
        objects = encoder.keys()
        x = (1,2,3,4,5,6)
        performance = [labels2.count(i) for i in encoder.values()]
        print(performance)
        import matplotlib.pyplot as plt
        plt.bar(x, performance, align='center', alpha=0.7)
        plt.xticks(x, objects)
        plt.ylabel('Post Number')
        plt.title(title)
        plt.show()
        #print(labels2)
        return labels2

    labels_tr = encode_label(labels_tr,"train")
    labels_te = encode_label(labels_te,"test")



    # 中文分词
    import jieba
    mywords = " ".join(jieba.cut(mytext))

    #生成词云，展示预处理结果
    text = " ".join(mywords)
    # from wordcloud import WordCloud
    # import os
    # import PIL.Image as image
    # import numpy as np
    # print("Generating word cloud")
    # wordcloud = WordCloud(
    #     font_path=os.getcwd() + "/font/kaiti.ttf",
    #     mask = np.array(image.open(os.getcwd() + "/font/map.jpg")),
    #     mode='RGBA',
    #     background_color='white',
    #     scale=5
    # ).generate(text)
    # image_produce = wordcloud.to_image()
    # link = os.getcwd() + "/font/wordcloud_after_processing.png"
    # wordcloud.to_file(link)
    # #image_produce.show()
    # print("word cloud has been genrated in "+str(link))
    # 保存分词
    f = open('splite_word_all.txt', 'w', encoding='utf-8')
    f.write(mywords)
    f.close()

    # 计算词向量
    from gensim.models import word2vec as w2v

    # 模型训练，生成词向量
    import os
    sentences_cut = w2v.LineSentence(os.getcwd()+"/"+sentence_cut_name)
    # vector size
    sample_length = 100
    print("sample_length",sample_length)
    model = w2v.Word2Vec(sentences_cut, vector_size=sample_length, window=15, min_count=5,
                         workers=4)  # 参数含义：数据源，生成词向量长度(30)，时间窗大小，最小词频数，线程数
    model.save(model_file_name)

    return sentences_tr,sentences_te,labels_tr,labels_te






