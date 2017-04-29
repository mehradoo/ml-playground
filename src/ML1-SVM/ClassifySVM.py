def classify(features_train, labels_train):
    from sklearn.svm import SVC
    clf = SVC(kernel="linear")
    # clf = SVC(kernel="rbf")
    return clf.fit(features_train, labels_train)
