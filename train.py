from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
import jieba
import numpy as np
import collections
import csv
import os
import keras
from xpinyin import Pinyin

mycwd = "C:\\Users\\DING FAN\\Desktop\\自然语言处理\\中文分类\\wordtovec\\code\\school"
os.chdir(mycwd)
chinese_punctuations = ['，', '。', '：', '；', '?', '？',
                        '（', '）','(',')', '「', '」', '！', '“', '”', '\n', ' ']  # 中文標點去除


def stopwordslist(filepath):
    # 獲取停用詞表
    stopwords = [line.strip() for line in open(
        filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def list2file(list, file):
    # 將list寫入文件
    fout = open(file, 'w', encoding='utf-8')
    for item in list:
        for i in item:
            fout.write(str(i) + ' ')
        fout.write('\n')
    fout.close()


def dict2file(dict, file):
    # 將dict寫入文件
    with open(file, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(dict.keys())
        w.writerow(dict.values())
    f.close()


# In[2]:
## 探索數據分析(EDA)
# 計算訓練資料的字句最大字數
## 準備數據
maxlen = 0
num_recs = 0
word_freqs = collections.Counter()
os.chdir(mycwd)
stopwords = stopwordslist('stopwords_Chinese.txt')
sourcefile = []
tokenset = set()
p = Pinyin()

for filename in os.listdir(mycwd):  # 按照檔後綴名稱標識獲取輸入文件
    if (filename[-6:] == '_1.txt'):
        sourcefile.append(filename)
for i in range(len(sourcefile)):
    fin = open(sourcefile[i], 'r', encoding='utf-8')
    fout_content = []
    for line in fin:
        tmp_line = []
        label, sentence = line.strip().split("\t")
        tokens = jieba.cut(sentence)  # cut方法為jieba分詞
        if (len(sentence) > maxlen):
            maxlen = len(sentence)
        for word in tokens:
            if ((word not in chinese_punctuations) and (word not in stopwords)):
                word_freqs[word] += 1
                pinyin=p.get_initials(word)
                tokenset.add(pinyin)
                tmp_line.append(pinyin)
        num_recs += 1
        fout_content.append(tmp_line)
    list2file(fout_content, sourcefile[i][:5] + '2.txt')
    fin.close()
print('max_len ', maxlen)
print(num_recs)
print('nb_words ', len(word_freqs))
print(word_freqs)
tokenlist = list(tokenset)
print(tokenset)
print(tokenlist)

n_class = 16
MAX_FEATURES = 400
MAX_SENTENCE_LENGTH = 21
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word_index = {w: c + 2 for c, w in enumerate(tokenset)}
word_index["PAD"] = 0
word_index["UNK"] = 1
index2word = {v: k for k, v in word_index.items()}
dict2file(word_index, 'word_to_int_tables.txt')  # 將token word與整數標號之間的關係寫入文件
print(index2word)

###########################################################################
X = np.empty(num_recs, dtype=list)
y = np.zeros(num_recs)
i = 0
# 讀取訓練資料，將每一單字以 dictionary 儲存
sourcefile = []
for filename in os.listdir(mycwd):  # 按照檔後綴名稱標識獲取輸入文件
    if (filename[-6:] == '_1.txt'):
        sourcefile.append(filename)
for w in range(len(sourcefile)):
    fin = open(sourcefile[w], 'r', encoding='utf-8')
    for line in fin:
        label, sentence = line.strip().split("\t")
        tokens = jieba.cut(sentence)  # cut方法為jieba分詞
        seqs = []
        for word in tokens:
            pinyin2 = p.get_initials(word)
            if ((pinyin2 in word_index)and (word not in chinese_punctuations) and (word not in stopwords)):
                seqs.append(word_index[pinyin2])
            #else:
                #seqs.append(word_index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
print(X.shape)
print(y.shape)

# In[3]:
# padding
print('Loading data...')
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
y = keras.utils.to_categorical(y, n_class)
print(X.shape)
print(y.shape)

#Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=14)
Xtrain = Xtest = X
ytrain = ytest = y

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 50
kernel_size = 3
filters = 64

print('Build model...')
model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
#add Embedding (250*128)嵌入层embedding
model.add(Embedding(vocab_size,
                    EMBEDDING_SIZE,
                    input_length=MAX_SENTENCE_LENGTH))
#31*128 字典数*特征数=3968
model.add(Dropout(0.1))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
#(128*3+1)*64=24640

model.add(GlobalMaxPooling1D())

model.add(Dense(HIDDEN_LAYER_SIZE))
model.add(Dropout(0.1))
model.add(Activation('relu'))
#全连接层 激活层
model.add(Dense(n_class, activation='softmax'))
# binary_crossentropy

model.summary()
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# training model
model.fit(Xtrain, ytrain,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          validation_data=(Xtest, ytest))
INPUT_SENTENCES = ['校长室', '陳心婷老師在哪？', '我想要找咨輔志工']
XX = np.empty(len(INPUT_SENTENCES), dtype=list)
# word to vector
i = 0
for sentence in INPUT_SENTENCES:
    words = jieba.cut(sentence)
    seq = []
    for word in words:
        pinyin = p.get_initials(word)
        if ((pinyin in word_index) and (word not in chinese_punctuations) and (word not in stopwords)):
            seq.append(word_index[pinyin])
        #else:
            #seq.append(word_index['UNK'])
    XX[i] = seq
    i += 1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
#labels = [int(round(np.argmax(x))) for x in model.predict(XX)]
labels = [None] * len(XX)
for index, l in enumerate(XX):
    ls = set(l)
    ls.remove(0)
    if len(ls) > 0:
        labels[index] = int(round(np.argmax(model.predict(np.array([l])))))
    else:
        labels[index] = 16

label2word = {0: '教務處以及教務處長室在行政樓一樓', 1: '該小姐/先生在行政一樓教務處', 2: '請前往註冊組，註冊組在行政一樓', 3: '該小姐/先生在行政一樓註冊組'
              , 4: '該小姐/先生在行政一樓課務組', 5: "請前往行政一樓課務組詢問相關規則，或前往課務組網頁"
              , 6: '該小姐/先生在行政一樓招生組', 7: '請前往行政一樓招生組詢問相關規則', 8: '生活輔導組在行政樓一樓', 9: '該小姐/先生在行政一樓生活輔導組', 10: '陳振遠校長在行政二樓校長室',
              11: '副校長室在行政二樓', 12: '影印部在宗教一樓，可以付費提供影印服務', 13: '衛生保健組在行政三樓，可以提供測升高體重，傷口應急處理的服務',
              14: '咨商辅导組在行政三樓，你可以找咨商辅导组的老师解决心中烦恼', 15: '你好！我是義大校務達人，你可以問我學生在行政大樓的相關事務哦', 16: '對不起，我不懂你在說什麼'}
# display
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))

model.save('school.h5')
