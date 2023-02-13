# returns Euclidean distance between vectors a dn b
import numpy as np
import k_nearest_neighbor as knn_class


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    return(labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    #show('valid.csv','pixels')
    dataset = read_data("train.csv")
    features_array = []
    targets_array = []
    for x in dataset:
        features_array.append(x[1])
        targets_array.append(x[0])
    features = np.array(features_array)
    targets = np.array(targets_array)
    newknn = knn_class.KNearestNeighbor(5,aggregator="mode")
    newknn.fit(features,targets)

    test_dataset = read_data("valid.csv")
    features_array = []
    targets_array = []
    for x in test_dataset:
        features_array.append(x[1])
        targets_array.append(x[0])
    test_features = np.array(features_array)
    test_targets = np.array(targets_array)
    z = newknn.predict(test_features)
    acc = 0
    for i in range(len(z)):
        if z[i] == test_targets[i]:
            acc += 1
    print("Accuracy :",acc*100/200)

if __name__ == "__main__":
    main()
    