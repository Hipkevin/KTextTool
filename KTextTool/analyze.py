from sklearn.feature_extraction.text import TfidfVectorizer
from .preProcess import prePro
import numpy as np

from gensim import corpora, models
from sklearn.metrics.pairwise import cosine_similarity


def TfIdf(words_ls, topN):
    """
    查看文档关键词
    :param words_ls: 语料
    :param topN: 前n个关键词
    :return: 输出关键词以及对应权重
    """
    proText = prePro(words_ls)
    corpus = []
    for t in proText:
        corpus.append(' '.join(t))

    corpus = [' '.join(corpus)]

    vectorize = TfidfVectorizer()
    x = vectorize.fit_transform(corpus).toarray()
    feature = vectorize.get_feature_names()

    keyWords = []
    score = []
    for m in x:
        top = np.argsort(m)[::-1][:topN]

        key = []
        s = []
        for t in top:
            key.append(feature[t])
            s.append(m[t])

        keyWords.append(key)
        score.append(s)
    return keyWords, score


def lda(words_ls, searchRange=10, topic=0, show=5, path=0):
    """
    lda分析
    :param words_ls: 语料
    :param searchRange: bestTopic搜索范围
    :param topic: 指定提取个数
    :param show: 主题向量词的个数
    :param path: 模型保存路径
    """
    def cosDistance(array):
        distance = cosine_similarity(np.mat(array))
        return distance.mean()

    def bestTopics(corpus, dictionary, searchRange=50):
        cosArray = []
        for k in range(searchRange):
            lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k + 1)

            array = []
            for topicArray in lda.get_topics():
                array.append(topicArray)

            cosArray.append(cosDistance(array))

        return np.argmin(cosArray)

    # 构造词典
    dictionary = corpora.Dictionary(words_ls)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus = [dictionary.doc2bow(words) for words in words_ls]

    if not topic:
        best = bestTopics(corpus=corpus, dictionary=dictionary, searchRange=searchRange)
        print('Best topics:', best)
    else:
        best = topic

    # lda模型，num_topics设置主题的个数
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=best)

    # 打印所有主题，每个主题显示5个词
    for topic in lda.print_topics(num_words=show):
        print(topic)

    if path:
        lda.save(path)