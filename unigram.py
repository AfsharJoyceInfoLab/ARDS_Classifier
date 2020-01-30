import pickle
import collections

from build_data import DataBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':

    args = {
        'corpus': '/NLPShare/ARDS/lessparsed/xin/',

        # tfidf
        'max_features': 1000,
        'features': '/NLPShare/ARDS/lessparsed/xin/features_without_phi.txt',

        # svc
        'kernel': 'linear',
        'random_state': 1,
        'probability': True,

        # grid search
        'search_space': {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
        'search_target': 'roc_auc',
        'cv': 10,
    }

    print('\n\n***Loading the data***\n\n')
    data_builder = DataBuilder(corpus=args['corpus'])
    examples_train, labels_train, examples_test, labels_test = data_builder.get_data()
    assert len(examples_train) == len(labels_train)
    assert len(examples_test) == len(labels_test)

    print('Finished loading.')
    print('Number of training examples = {}'.format(len(labels_train)))
    print('positive examples in train = {}'.format(collections.Counter(labels_train)[1]))
    print('negative examples in train = {}'.format(collections.Counter(labels_train)[0]))
    print('Number of test examples = {}'.format(len(labels_test)))
    print('positive examples in test = {}'.format(collections.Counter(labels_test)[1]))
    print('negative examples in test = {}'.format(collections.Counter(labels_test)[0]))

    # get features
    print('***Convert text to features***')
    tfidf = TfidfVectorizer(strip_accents='unicode',
                            lowercase=True,
                            analyzer='word',
                            stop_words='english',
                            norm='l2',
                            use_idf=True,
                            smooth_idf=True,
                            ngram_range=(1, 1),
                            max_features=args['max_features'],
                            vocabulary=args['features'])
    tfidf.fit(examples_train)
    print('Number of features = {}'.format(len(tfidf.vocabulary_)))

    features_train = tfidf.transform(examples_train)
    features_test = tfidf.transform(examples_test)

    # train svc
    print('***tune and train the model***')
    svc = SVC(kernel="linear", random_state=1, probability=True)
    grid_search = GridSearchCV(estimator=svc,
                               param_grid=args['search_space'],
                               scoring=args['search_target'],
                               cv=args['cv'],
                               n_jobs=-1)
    grid_search.fit(features_train)
    grid_search.fit(features_test)
    print('The best hyper-parameters are')
    print(grid_search.best_params_)
    svc.set_params(**grid_search.best_params_)

    svc.fit(X=features_train,
            y=labels_train)

    # performance
    print('\n\n***Eval Train***\n')
    pred_labels_train = svc.predict(features_train)
    pred_probs_train = svc.predict_proba(features_train)
    roc_auc_train = metrics.roc_auc_score(y_true=labels_train,
                                          y_score=pred_probs_train[:, 1])
    print('ROC AUC = {}'.format(roc_auc_train))
    print(metrics.classification_report(y_true=labels_train,
                                        y_pred=pred_labels_train))

    print('\n\n***Eval Test***\n')
    pred_labels_test = svc.predict(features_test)
    pred_probs_test = svc.predict_proba(features_test)
    roc_auc_test = metrics.roc_auc_score(y_true=labels_test,
                                          y_score=pred_probs_test[:, 1])
    print('ROC AUC = {}'.format(roc_auc_test))
    print(metrics.classification_report(y_true=labels_test,
                                        y_pred=pred_labels_test))



