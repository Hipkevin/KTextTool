import re
import jieba
from gensim import corpora, models
from sklearn import preprocessing


def prePro(content):
    # 去字符
    formatList = list(map(lambda a: re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]").sub('', str(a)), content))

    # 分词
    cut_words = list(map(lambda s: jieba.lcut(s), formatList))  # lambda表达式和map结合速度更快

    # 停词
    filepath = "KTextTool/chineseStopWords.txt"
    stoplist = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    stopwords = dict(zip(stoplist, stoplist))
    contents_clean = []
    for line in cut_words:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
        contents_clean.append(line_clean)

    return contents_clean


def doc2lsi(content, label=[], dimension=20, flag=False):
    # 得到lsi向量
    word_list = prePro(content)
    dictionary = corpora.Dictionary(word_list)
    new_corpus = [dictionary.doc2bow(text) for text in word_list]
    tfidf = models.TfidfModel(new_corpus)
    tfidf_vec = []
    for i in range(len(word_list)):
        string_bow = dictionary.doc2bow(word_list[i])
        string_tfidf = tfidf[string_bow]
        tfidf_vec.append(string_tfidf)

    # 使每个句子的维度一致
    lsi_model = models.LsiModel(corpus=tfidf_vec, id2word=dictionary, num_topics=dimension)
    lsi_vec = []
    for i in range(len(word_list)):
        string_bow = dictionary.doc2bow(word_list[i])
        string_lsi = lsi_model[string_bow]
        lsi_vec.append(string_lsi)

    # sklearn使用的类型为array,必须转换
    from scipy.sparse import csr_matrix
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in lsi_vec:
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    lsi_sparse_matrix = csr_matrix((data, (rows, cols)))  # 稀疏向量
    lsi = lsi_sparse_matrix.toarray()  # 密集向量

    # 标签转化为array
    encoder = preprocessing.LabelEncoder()
    lab = encoder.fit_transform(label)

    if flag:
        return lsi, lab
    else:
        return lsi