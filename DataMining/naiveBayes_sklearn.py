from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

train_data = ['Chinese Beijing Chinese', 'Chinese Japan Beijing', 'Chinese Beijing Tokyo', 'Tokyo Japan Chinese', 'Japan Japan Beijing','Tokyo Japan Chinese']
train_labels = ['c', 'c', 'c', 'j', 'j', 'j']

vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_data)

clf = MultinomialNB()

clf.fit(train_vectors, train_labels)
priors = clf.class_log_prior_
feature_names = vectorizer.get_feature_names_out()
for i, class_name in enumerate(clf.classes_):
    print("类别{}的条件概率为：".format(class_name))
    for j, feature_name in enumerate(feature_names):
        print("\t{}:{}".format(feature_name, clf.feature_log_prob_[i][j]))
print(clf.class_log_prior_)  # 输出先验概率
print(clf.feature_log_prob_)  # 输出后验概率

"""由于样本缺少，造成分类并不准确"""


# 对西瓜数据集的处理
data = list()
