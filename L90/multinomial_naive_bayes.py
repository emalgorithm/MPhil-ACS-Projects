import numpy as np

class MultinomialNaiveBayes():
    def __init__(self, classes, num_feat, smoothing_value=0):
        # Number of features the model uses
        self.num_feat = num_feat
        # List of the classes
        self.classes = classes
        # Dictionary mapping each class to the prior probability p(C=c)
        self.class_to_prior = {c: 0 for c in classes}
        # self.class_to_feature_to_cond_prob[c][x] is used to store the estimate of the conditional probability p(X=x|C=c)
        self.class_to_feature_to_cond_prob = {c: np.zeros((num_feat,)) for c in classes}
        # A smoothing value of 0 is equivalent to no smoothing
        self.smoothing_value = smoothing_value
        
    def fit(self, X, y):
        y = np.array(y)
        X = np.array(X)
        # Computer priors
        for c in y:
            self.class_to_prior[c] += 1
        self.class_to_prior.update({c: self.class_to_prior[c] / len(y) for c in self.classes})
        
        # Compute estimate of the conditional probability p(X=x|C=c)
        for c in self.classes:
            X_c = X[y == c]
            features_frequencies = np.sum(X_c, axis=0)
            self.class_to_feature_to_cond_prob[c] = (features_frequencies + self.smoothing_value) / sum(features_frequencies + self.smoothing_value)
        
    def predict(self, X):
        return np.argmax(np.stack([self.compute_scores(X, c) for c in self.classes], axis=-1), axis=1)
    
    def compute_scores(self, X, c):
        # If smoothing is not applied, some conditional probability will be zero and so we can't take the log of them
        if self.smoothing_value == 0:
            scores = []
            log_cond_prob = np.log(self.class_to_feature_to_cond_prob[c])
            for x in X:
                score = np.log(self.class_to_prior[c])
                for i, count in enumerate(x):
                    if count != 0:
                        score += log_cond_prob[i] * count
                scores.append(score)
            return scores
            
        adjusted_cond_prob = np.array([p if p != 0 else 10 ** -100 for p in self.class_to_feature_to_cond_prob[c]])
        return np.log(self.class_to_prior[c]) + np.matmul(X, np.log(adjusted_cond_prob))

#     def compute_score(self, x, c):
#         # Compute score (unnormalized log probability) for given class
#         return np.log(self.class_to_prior[c]) + np.dot(x, np.log(self.smooth(self.class_to_feature_to_cond_prob[c])))