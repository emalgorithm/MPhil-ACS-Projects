import numpy as np
import os
from collections import Counter
import math
import scipy

def process_file(filepath):
    """Given a file, returns a list of tokens for that file"""
    x = []
    with open(filepath, 'r') as f:
        for l in f:
            # Filter lines which consist only of new line operator
            if l == '\n':
                continue
            
            token, pos_tagging = l.split('\t')
            x.append(token)
    return x

def preprocess_data(datapath, sentiment='POS'):
    idx = 0
    X = []
    y = []
    sentiment_value = 1 if sentiment == 'POS' else 0
    
    # For file in the folder
    current_path = datapath + sentiment
    for f in sorted(os.listdir(current_path)):
        x = process_file(current_path + '/' + f)
        X.append(x)
        y.append(sentiment_value)

    return X, y

def get_unigram_dictionary(X, cutoff=1):
    token_counter = Counter(np.concatenate(X))
    idx = 0
    token_to_idx = {}
    
    for x in X:
        for token in x:
            if token_counter[token] >= cutoff and token not in token_to_idx:
                token_to_idx[token] = idx
                idx += 1
                
    return token_to_idx

def get_bigram_dictionary(X, cutoff=1, token_to_idx={}):
    X_bigram = []
    for x in X:
        X_bigram += [(x[i], x[i + 1]) for i, _ in enumerate(x) if i < len(x) - 1 ]

    token_counter = Counter(X_bigram)
    idx = len(token_to_idx)
    
    for x in X:
        x_bigram = [(x[i], x[i + 1]) for i, _ in enumerate(x) if i < len(x) - 1 ]
        for token in x_bigram:
            if token_counter[token] >= cutoff and token not in token_to_idx:
                token_to_idx[token] = idx
                idx += 1
                
    return token_to_idx

def get_dictionary(X, unigram_cutoff=1, bigram_cutoff=1, unigram=True, bigram=False):
    """
    Returns a dictionary which maps each token to its index in the feature space.
    Tokens which appear less times than specified by the cutoff are discarded
    """
    token_to_idx = {}
    if unigram:
        token_to_idx = get_unigram_dictionary(X, unigram_cutoff)
    if bigram:
        token_to_idx = get_bigram_dictionary(X, bigram_cutoff, token_to_idx)
    
    print("Generated {} features".format(len(token_to_idx)))
                    
    return token_to_idx

def featurize_data(X, token_to_idx):
    """Convert each sample from a list of tokens to a multinomial bag of words representation"""
    X_unigram_and_bigram = []
    for x in X:
        X_unigram_and_bigram.append(x + [(x[i], x[i + 1]) for i, _ in enumerate(x) if i < len(x) - 1 ])
        
    X_feat = []
    for x in X_unigram_and_bigram:
        x_feat = np.zeros((len(token_to_idx)))
        for token in x:
            if token in token_to_idx:
                x_feat[token_to_idx[token]] += 1
        X_feat.append(x_feat)
    
    return X_feat

def sign_test(y1_pred, y2_pred, y_test):
    # plus counts the number of times model1 beats model2
    plus = 0
    # minus counts the number of times model2 beats model1
    minus = 0
    # nul counts the number of times model1 and model2 tie    
    null = 0
    
    for i in range(len(y_test)):
        correct1 = 1 if y1_pred[i] == y_test[i] else 0
        correct2 = 1 if y2_pred[i] == y_test[i] else 0
        
        if correct1 > correct2:
            plus += 1
        elif correct2 > correct1:
            minus += 1
        else:
            null += 1
          
    # If we have too many datapoints, than our custom method overflows
    # Therefore, in that case we use scipy function
    if len(y_test) > 500:
        return scipy.stats.binom_test(plus + null // 2, len(y_test))
    
    p = 0
    N = 2 * math.ceil(null / 2) + plus + minus
    k = math.ceil(null / 2) + min(plus, minus)
    q = 0.5
    
    for i in range(k):
        p += 2 * (q ** i) * ((1 - q) ** (N - i)) * (math.factorial(N) / (math.factorial(i) * math.factorial(N - i))) 
        
    print("p: {}".format(p))   
    
def cross_validation(model, X, y, k=10, unigram=True, bigram=False, unigram_cutoff=1, bigram_cutoff=1):
    # Split indexes
    idxs = np.array(range(len(y)))
    
    folds_idxs = [[] for _ in range(k)]
    for idx in idxs:
        fold = idx % k
        folds_idxs[fold].append(idx)
        
    # Run test
    accuracies = []
    num_of_feat = []
    total_y_pred = []
    total_y_test = []
    
    for test_fold in range(k):
        print("Running iteration {} out of {} of cross validation".format(test_fold + 1, k))
        test_idxs = folds_idxs[test_fold]
        train_idxs = list(set(np.concatenate(folds_idxs)) - set(test_idxs))
        X_train = X[train_idxs]
        y_train = y[train_idxs]
        
        X_test = X[test_idxs]
        y_test = y[test_idxs]
        
        token_to_idx = get_dictionary(X_train, unigram_cutoff=unigram_cutoff, bigram_cutoff=bigram_cutoff, unigram=unigram, bigram=bigram)
        X_train = featurize_data(X_train, token_to_idx)
        X_test = featurize_data(X_test, token_to_idx)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        total_y_pred = np.concatenate([total_y_pred, y_pred])
        total_y_test = np.concatenate([total_y_test, y_test])
        n_correct = sum(1 for i, _ in enumerate(y_pred) if y_pred[i] == y_test[i])
        accuracy = n_correct * 100 / len(X_test)
        accuracies.append(accuracy)
        num_of_feat.append(len(token_to_idx))
        
        print("{0:.2f}% of sentences are correctly classified \n".format(accuracy))
        
    print("Finished running {}-fold cross validation".format(k))
    print("Average number of features: {}".format(np.mean(num_of_feat)))
    print("Accuracy is {}(+- {})\n".format(np.mean(accuracies), np.std(accuracies)))
    
    return total_y_pred, total_y_test