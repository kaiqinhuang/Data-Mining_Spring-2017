import numpy as np

def main():
    att_data = np.loadtxt(open("spamdata_binary.csv", "rb"), delimiter=",", dtype = 'int')
    class_data = np.loadtxt(open("spamlabels.csv", "rb"), dtype = 'int')

    rand_ordering = np.random.permutation(len(att_data))
    att_data = att_data[rand_ordering]
    class_data = class_data[rand_ordering]
    
    n_folds = 10
    print("Number of Folds: " , n_folds)
    accuracies = cross_validate(att_data, class_data, n_folds)
    print("Average accuracy: " , sum(accuracies)/len(accuracies))

def cross_validate(att_data, class_data, n_folds):
    split_att_data = np.array_split(att_data, n_folds)
    split_class_data = np.array_split(class_data, n_folds)
    return [fold_accuracy(split_att_data, split_class_data, i) for i in range(n_folds)]

def fold_accuracy(split_att_data, split_class_data, fold_index):
    (train_att_data, test_att_data) = group_test_train(split_att_data, fold_index)
    (train_class_data, test_class_data) = group_test_train(split_class_data, fold_index)
    classifier = NaiveBayesClassifier(train_att_data, train_class_data)
    return classifier_accuracy(classifier, test_att_data, test_class_data)

def classifier_accuracy(classifier, test_att_data, test_class_data):
    n_correct = 0
    for i in range(len(test_att_data)):
        if classifier.classify(test_att_data[i]) == test_class_data[i]:
            n_correct += 1
    return n_correct/len(test_att_data)

def group_test_train(split_arr, test_index):
    test_arr = split_arr[test_index]
    train_arr = np.concatenate(split_arr[:test_index] + split_arr[test_index+1:])
    return (train_arr, test_arr)

#Assumes binary data
class NaiveBayesClassifier(object):
    def __init__(self, attribute_data, classifications):
        self.attribute_data = attribute_data
        self.classifications = classifications
        self.epsilon = 0.0000001
        self.train()

    def train(self):
        self.set_log_priors()
        self.set_cond_log_likelihoods()

    def set_log_priors(self):
        counts = [np.count_nonzero(self.classifications == val) for val in [0,1]]
        self.log_priors = [np.log2(count/len(self.classifications)) for count in counts]
        
    def set_cond_log_likelihoods(self):
        self.cond_log_likelihoods = [self.cond_log_likelihoods(class_value) for class_value in [0,1]]

    def cond_log_likelihoods(self, class_value):
        subset = self.attribute_data[self.classifications == class_value]
        proportion_1 = sum(subset)/len(subset)
        return [np.log2(1 - proportion_1 + self.epsilon),
                np.log2(proportion_1 + self.epsilon)]
    
    def classify(self, att_row):
        log_like_0 = self.log_priors[0] + self.attributes_cond_likelihood(att_row,0)
        log_like_1 = self.log_priors[1] + self.attributes_cond_likelihood(att_row,1)
        if log_like_0 > log_like_1:
            return 0
        else:
            return 1
        
    def attributes_cond_likelihood(self, att_row, classification):
        return (np.dot((1-att_row), self.cond_log_likelihoods[classification][0]) +
                np.dot(att_row,     self.cond_log_likelihoods[classification][1]))

if __name__ == "__main__":
    main()
