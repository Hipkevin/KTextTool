# KTextTool

## 1.功能介绍
    用于基础的机器学习&文本分析
    完成基本的分类和回归任务
    对文本进行简单的分析与挖掘

## 2.依赖
    python3.8
    numpy
    pandas
    matplotlib
    seaborn
    skearn
    xgboost
    gensim
    imblearn
    jieba

## 3.项目结构
    .
    │  README.md
    │
    └─KText
        │  analyze.py
        │  chineseStopWords.txt
        │  FE.py
        │  preProcess.py
        └─  __init__.py

## 4.使用方法
### 4.1 preProcess模块
    prePro(content)
    doc2lsi(content, label=[], dimension=20, flag=False)

    content输入格式如下：
    content = [
        ["sentence_1"],
        ["sentence_2"],
        ["sentence_3"]
    ]  # 文本矩阵

### 4.1 analyze模块
    TfIdf(words_ls, topN)
    lda(words_ls, searchRange=10, topic=0, show=5, path=0)
    
    words_ls输入格式如下：
    words_ls = [
        ["word_11", "word_12"],
        ["word_21", "word_22", "word_23"],
        ["word_31"]
    ]  # prePro处理后的词表

## Others
    @Author: Kevin
