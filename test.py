from keras.preprocessing import sequence
from keras.models import load_model
import jieba
import numpy as np
from aip import AipSpeech
import voice.sound as sound
import time
from tf.langconv import Converter
from xpinyin import Pinyin


APP_ID = '14865501'
API_KEY = 'rtk9bvXgRpBhIU2WX7NSbCv8'
SECRET_KEY = 'byhh0an1FQ5VGDB1UgkX0HjQRYuC1yQu'

p = Pinyin()

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

model = load_model('school.h5')

chinese_punctuations = ['，', '。', '：', '；', '?', '？',
                                '（', '）', '「', '」', '！', '“', '”', '\n', ' ']  # 中文標點去除

def stopwordslist(filepath):
    # 獲取停用詞表
    content = [line.strip() for line in open(
        filepath, 'r', encoding='utf-8').readlines()]
    return content

def getdata(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        cnt = 0
        for line in f:
            if(cnt==0):
                tmpline=line.replace("\n","")
                tmpkey=tmpline.split(',')
                cnt += 1
            else:
                tmpline=line.replace("\n","")
                tmpval=tmpline.split(',')
                tmpval = [ int(x) for x in tmpval ]
        content =dict(zip(tmpkey,tmpval))
    f.close()
    return content

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def simple2tradition(line):
    #将简体转换成繁体
    line = Converter('zh-hant').convert(line)
    return line

word_index=getdata('word_to_int_tables.txt')
word_index2=getdata('word_to_int_tables2.txt')
word_index.update(word_index2)
stopwords = stopwordslist('stopwords_Chinese.txt')
client.asr(get_file_content('question.wav'), 'wav', 8000, dict(dev_pid=1536))

while True:
    sound.rec("question.wav")
    cmd = client.asr(get_file_content('question.wav'), 'wav', 8000, dict(dev_pid=1536))
    if cmd["err_msg"] != 'success.':
        print('沒有收到語音訊息')
        time.sleep(4)
    if cmd["err_msg"] == 'success.' and cmd["result"] == ['谢谢']:
        break
    if cmd["err_msg"] == 'success.':
        INPUT_SENTENCES =cmd["result"]
        str=INPUT_SENTENCES[0]
        print(simple2tradition(str))
        #print(INPUT_SENTENCES)
        XX = np.empty(len(INPUT_SENTENCES), dtype=list)
# word to vector
        i = 0
        for sentence in INPUT_SENTENCES:
            words = jieba.cut(sentence)
            seq = []
            for word in words :
                pinyin = p.get_initials(word)
                if ((pinyin in word_index)and (word not in chinese_punctuations) and (word not in stopwords)):
                    seq.append(word_index[pinyin])
                    print(pinyin)
                #else:
                    #seq.append(word_index['UNK'])
            XX[i] = seq
            i += 1

        XX = sequence.pad_sequences(XX, maxlen=21)
        for index, l in enumerate(XX):
            ls = set(l)
            ls.remove(0)
            if len(ls) > 0:
                labels = int(round(np.argmax(model.predict(np.array([l])))))
            else:
                labels = 16

        label2word = {0: '教務處以及教務處長室在行政樓一樓', 1: '該小姐/先生在行政一樓教務處', 2: '請前往註冊組，註冊組在行政一樓', 3: '該小姐/先生在行政一樓註冊組'
              , 4: '該小姐/先生在行政一樓課務組', 5: "請前往行政一樓課務組詢問相關規則，或前往課務組網頁"
              , 6: '該小姐/先生在行政一樓招生組', 7: '請前往行政一樓招生組詢問相關規則', 8: '生活輔導組在行政樓一樓', 9: '該小姐/先生在行政一樓生活輔導組', 10: '陳振遠校長在行政二樓校長室',
              11: '副校長室在行政二樓', 12: '影印部在宗教一樓，可以付費提供影印服務', 13: '衛生保健組在行政三樓，可以提供測升高體重，傷口應急處理的服務',
              14: '咨商辅导組在行政三樓，你可以找咨商辅导组的老师或志工解决心中烦恼', 15: '你好！我是義大校務達人，你可以問我學生在行政大樓的相關事務哦', 16: '對不起，我不懂你在說什麼'}
        # display
        print('{}'.format(label2word[labels]))

        result = client.synthesis(label2word[labels], 'zh', 1, {
            'vol': 5, 'per': 4, 'aue':6
        })

        if not isinstance(result, dict):
            with open('audio.wav', 'wb') as f:
                f.write(result)
            f.close()

        sound.play()
        time.sleep(2)