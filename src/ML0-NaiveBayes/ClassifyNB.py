def classify(features_train, labels_train):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()

    # print features_train
    # print labels_train

    return gnb.fit(features_train, labels_train)
