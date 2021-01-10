import pandas as pd
import numpy as np


def showClassificationData(X, Y, ifFit=False, overSampling='None'):
    """
    showData实现多维二分类任务的数据可视化，函数接受数值型变量，使用PCA可视化。
    :param overSampling: 对样本进行重采样
    :param ifFit: 画图时是否对数据进行拟合
    :param X: 设计矩阵(不含标签)
    :param Y: 分类标签
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN

    def getSMOTE():
        return SMOTE(random_state=10)

    def getSMOTETomek():
        return SMOTETomek(random_state=10)

    def getSMOTEENN():
        return SMOTEENN(random_state=10)

    switch = {'SMOTE': getSMOTE,
              'SMOTETomek': getSMOTETomek,
              'SMOTEENN': getSMOTEENN}

    if overSampling != 'None':
        overSampler = switch.get(overSampling)()
        X, Y = overSampler.fit_sample(X, Y)

    # PCA提取两个主成分
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    X = pd.concat([X, Y], ignore_index=True, axis=1)
    X.columns = ['firstIngridient', 'secondIngridient', 'label']
    sns.lmplot('firstIngridient', 'secondIngridient', X, hue='label', fit_reg=ifFit)
    plt.show()


def showFeature(X, ifFit=False, method='PCA'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest

    if method == 'PCA' or 'pca':
        # PCA提取两个主成分
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
    if method == 'KBest':
        selector = SelectKBest(k=2)
        X = selector.fit_transform(X)

    X = pd.DataFrame(X)

    X.columns = ['firstIngridient', 'secondIngridient']
    sns.lmplot('firstIngridient', 'secondIngridient', X, fit_reg=ifFit)
    plt.show()


def showHistogram(Y, ifLog=False):
    """
    showHistgoram实现对计数变量的分布可视化
    :param ifLog: 是否对数据进行对数变换
    :param Y: 响应变量
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    Y = pd.DataFrame(Y)
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    Y.hist(ax=ax, bins=100)
    if ifLog:
        ax.set_yscale('log')
    ax.tick_params(labelsize=14)
    ax.set_xlabel('count', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    plt.show()


def binaryzation(Y, median=0):
    """
    binaryzation实现计数的二值化，将中位数设置为阈值，大于阈值为1
    :param median: 二值化阈值
    :param Y: 计数
    :return: 返回二值化标记的标签
    """
    if median == 0:
        median = np.median(Y)  # 默认使用中位数作为阈值
    else:
        if median > np.max(Y):  # 超出上限则将阈值设置为上限
            median = np.max(Y)

    label = []
    for l in Y:
        if l >= median:
            label.append(1)
        else:
            label.append(0)

    return label


def multiClassificationModel(X, Y, testSize=0.3, cv=5, ifSmote=False):
    """
    multiClassificationModel实现朴素贝叶斯、支持向量机、Logistic回归和XGBoost在数据集上的分类性能测试
    :param ifSmote: 是否做SMOTE重采样
    :param X: 特征矩阵
    :param Y: 标签
    :param testSize: 测试集大小，默认为0.3
    :param cv: k折交叉验证的k值
    """
    from sklearn import naive_bayes, svm, linear_model
    from sklearn.model_selection import train_test_split, cross_val_score
    import xgboost
    from imblearn.over_sampling import SMOTE

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=testSize, random_state=1)

    if ifSmote:
        smo = SMOTE(random_state=10)
        x_train, y_train = smo.fit_sample(x_train, y_train)

    print("----训练集accuracy----")
    # 朴素贝叶斯
    NB = naive_bayes.GaussianNB().fit(x_train, y_train)
    print("NB   : ", NB.score(x_train, y_train))

    # SVM
    SVM = svm.SVC().fit(x_train, y_train)
    print("SVM   : ", SVM.score(x_train, y_train))

    # logisticRegression
    LR = linear_model.LogisticRegression().fit(x_train, y_train)
    print("LR    : ", LR.score(x_train, y_train))

    # xgboost
    Xgb = xgboost.XGBClassifier().fit(x_train, y_train)
    print("Xgb   : ", Xgb.score(x_train, y_train))

    print("----交叉验证accuracy----")
    scores = cross_val_score(NB, X, Y, cv=cv)
    print("NB    :", scores.mean())

    scores = cross_val_score(SVM, X, Y, cv=cv)
    print("SVM    :", scores.mean())

    scores = cross_val_score(LR, X, Y, cv=cv)
    print("LR    :", scores.mean())

    scores = cross_val_score(Xgb, X, Y, cv=cv)
    print("Xgb    :", scores.mean())


def regressionKFold(model, X, Y, cv=5):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv, shuffle=True)

    score = []
    for train, test in kf.split(X, Y):
        model.fit(X[train], Y[train])
        s = model.score(X[test], Y[test])
        score.append(s)

    return np.mean(score)


def multiRegressionModel(X, Y, testSize=0.3, cv=5):
    """
        multiModel实现朴素贝叶斯、支持向量机、Logistic回归和XGBoost在数据集上的性能测试
        :param X: 特征矩阵
        :param Y: 标签
        :param testSize: 测试集大小，默认为0.3
        :param cv: k折交叉验证的k值
        """
    from sklearn import svm, linear_model
    from sklearn.model_selection import train_test_split, cross_val_score
    import xgboost

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=testSize, random_state=1)

    print("----训练集accuracy----")
    # SVM
    SVM = svm.SVR().fit(x_train, y_train)
    print("SVM   : ", SVM.score(x_train, y_train))

    # linearRegression
    LR = linear_model.LinearRegression().fit(x_train, y_train)
    print("LR    : ", LR.score(x_train, y_train))

    # xgboost
    Xgb = xgboost.XGBRegressor().fit(x_train, y_train)
    print("Xgb   : ", Xgb.score(x_train, y_train))

    print("----交叉验证accuracy----")
    scores = cross_val_score(SVM, X, Y, cv=cv).mean()
    print("SVM    :", scores)

    scores = cross_val_score(LR, X, Y, cv=cv).mean()
    print("LR    :", scores)

    scores = cross_val_score(Xgb, X, Y, cv=cv).mean()
    print("Xgb    :", scores)


def save2excel(X, Y):
    """
    保存结构数据
    :param X: 设计矩阵
    :param Y: label
    """
    sample = pd.DataFrame(np.column_stack((X, Y)))

    writer = pd.ExcelWriter('data.xlsx')  # 写入Excel文件
    sample.to_excel(writer, 'sheet1')
    writer.save()
    writer.close()