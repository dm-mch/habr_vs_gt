#!/usr/bin/python

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import argparse

from grabber import download

class Classifier:
    """
    Classifier text for categories

    """
    def __init__(self, name="model"):
        self.name = name
        
    def train(self, files, test_size=0.25):
        """
        Train model from files - json files, each file one category 

        """
        corpus, Y = self._load_corpus(files)
        X = self._get_features(corpus)
        if test_size > 0:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
            print("Train for validate on {} dataset...".format(1-test_size))
            clf = MultinomialNB().fit(X_train, Y_train)
            _y = clf.predict_proba(X_test)[:,1]
            #print(_y[:10])
            self._print_metrics(Y_test, _y)

        # for input predict train on full dataset
        print("Train for full corpus...")
        self.clf = MultinomialNB().fit(X, Y)
        self._print_main_features()
        self.save()

    def predict(self, file):
        """
        Predict category of text from file

        """
        if not hasattr(self, 'clf'):
            self.load()
        with open(file) as f:
            data = f.read()
            x = self.tfv.transform([data])
            p = self.clf.predict_proba(x)[0]
            indx = np.argmax(p)
            print("File %s is from %s category with %2.2f%%"%(file, self.categoris[indx], max(p) * 100))
            return p

    def save(self, filepath=None):
        if hasattr(self, 'clf'):
            pickle.dump([self.clf, self.categoris, self.tfv], open(filepath or self.name + '.p', "wb"))
            print("Model saved")
        else:
            raise Exception("Classifier not trained")

    def load(self, filepath=None):
        self.clf, self.categoris, self.tfv = pickle.load(open(filepath or self.name + '.p', "rb"))
        print("Model loaded")

    def _print_main_features(self):
        if hasattr(self, 'clf') and hasattr(self, 'voc'):
            probs = self.clf.feature_log_prob_
            for i,prob in enumerate(probs):
                print(self.categoris[i], ':', "\nMin prob:", ', '.join(self.voc[np.argsort(prob)[:20]]), "\nMax prob:", ', '.join(self.voc[np.argsort(prob)[-20:]]))

    def _print_metrics(self, y_true, y_score):
        roc_auc = metrics.roc_auc_score(y_true, y_score)
        print("roc_auc:", roc_auc)
        y_pred = np.rint(y_score)
        print("accuracy:", metrics.accuracy_score(y_true, y_pred))
        print("recall:", metrics.recall_score(y_true, y_pred))
        print("precision:", metrics.precision_score(y_true, y_pred))
        self._plot_roc(y_true, y_score, roc_auc)


    def _plot_roc(self, y_true, y_score, roc_auc):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.2])
        plt.ylim([-0.1,1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        name = self.name + "_roc_curve.png"
        plt.savefig(name)
        print("File %s saved"%(name))

    def _get_features(self, corpus):
        print("Transform corpus...")
        self.voc = self._get_vocabulary(corpus)
        self.tfv = TfidfVectorizer(vocabulary=self.voc)
        return self.tfv.fit_transform(corpus)


    def _get_vocabulary(self, corpus):
        cv = CountVectorizer()
        x = cv.fit_transform(corpus)
        counts = np.array(x.sum(axis=0))[0]
        sort_i = np.argsort(counts)
        features = np.array(cv.get_feature_names())
        all_size = len(sort_i)
        return features[sort_i[int(all_size * 0.833):-int(all_size * 0.018)]]

    def _load_corpus(self, files):
        self.categoris = [os.path.basename(f) for f in files]
        corpus = []
        y = []
        for i,f in enumerate(files):
            print("Parse...", f)
            with open(f, 'r') as fi:
                data = json.load(fi)
                for d in data:
                    corpus.append(d['title'] + ' ' + d['text'])
                    y.append(i)
        return corpus, y


def main():
    parser = argparse.ArgumentParser(description="Classifier text from habr and geektimes")

    parser.add_argument("-tr","--train", action="store_true", default = False, help = "train model")
    parser.add_argument("-test","--test", action="store_true", default = False, help = "test model")
    parser.add_argument("-f","--files", type=str, nargs='+', default = ["habrahabr.json", "geektimes.json"])
    parser.add_argument("-m","--model", type=str, default = "model")
    parser.add_argument("-ts","--testsize", type=int, default = 25, help = "test size for validate in percent")
    parser.add_argument("-d","--download", type=int, default = 0, help = "how many article download from habr and geektimes")

    args = parser.parse_args()

    c = Classifier(name=args.model)

    if args.download > 0:
        assert(args.download <= 10000)
        assert(len(args.files) == 2)
        download(args.download, habr=args.files[0], gt=args.files[1])

    if args.train:
        assert(len(args.files) == 2)
        assert(0 < args.testsize <= 100)
        c.train(args.files, test_size = args.testsize/100)
    elif args.test:
         c.load()
         for f in args.files:
            c.predict(f)
    else:
        print("""Usage example:
    habrgeek.py ---download 1000 --train
    habrgeek.py --train -f habrahabr.json geektimes.json
    habrgeek.py --test -f input.txt
    habrgeek.py ---download 1000
             """)
        parser.print_help()


if __name__ == "__main__":
    main()