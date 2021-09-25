import jionlp as jio

# 初始化应用
youdao_free_api = jio.YoudaoFreeApi()
youdao_api = jio.YoudaoApi(
        [{'appid': '4f490dca0215d784',
          'app_secret': 'n9Na3ZhZeeub0IOo4BTckpbf9QbFlqie'}])

def back_trans(text):
    trans_youdao = youdao_api(text=text,from_lang='zh-CHS', to_lang='en') #中译英
    back_trans_yy = youdao_api(text=trans_youdao,from_lang='en', to_lang='zh-CHS') #英译中
    return back_trans_yy

def back_trans_enhance(text):
    #   支持语言：中文(zh-CHS)、英文(en)、日文(ja)、法文(fr)、西班牙语(es)、
    #             韩文(ko)、葡萄牙文(pt)、俄语(ru)、德语(de)
    #print(youdao_api.__doc__)
    #trans_baidu  = Baidu_api(text=text,from_lang='zh',to_lang='en')  # 使用接口做单次调用
    trans_youdao = youdao_api(text=text,from_lang='zh-CHS', to_lang='ja')
    back_trans_yy = youdao_api(text=trans_youdao,from_lang='ja', to_lang='zh-CHS')

    trans_youdao_fr = youdao_api(text=text, from_lang='zh-CHS', to_lang='fr')
    back_trans_fr = youdao_api(text=trans_youdao_fr, from_lang='fr', to_lang='zh-CHS')

    trans_youdao_de = youdao_api(text=text, from_lang='zh-CHS', to_lang='de')
    back_trans_de = youdao_api(text=trans_youdao_de, from_lang='de', to_lang='zh-CHS')


    return back_trans_yy,back_trans_fr,back_trans_de
