
import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


training_data = np.genfromtxt('../dataset/dataset.csv',delimiter=',', dtype=np.int32)
#print(training_data)
input = training_data[:,:-1]

label = training_data[:, -1]

#Create Logistic Regression Classifier
lr_classifier = LogisticRegression()

crv = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in crv.split(input):
    # print("Train Index: ", train_index, "\n")
    # print("Test Index: ", test_index)
    # Divide dataset for training and testing
    train_input = input[train_index]
    test_input = input[test_index]

    train_output = label[train_index]
    test_output = label[test_index]

    # Train the classifier
    lr_classifier.fit(train_input, train_output)

    # Prediction
    lr_predicter = lr_classifier.predict(test_input)

    class_names = ['Phishing', 'Ligitimate']


    # Accuracy of the model
    accuracy = 100.0 * accuracy_score(test_output, lr_predicter)

    print("The accuracy of your Logistic Regression on testing data is: " + str(accuracy))

    #Plot confusion matrix
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_output, lr_predicter)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')
    # plt.show()

    #Average Precision call
    # average_precision = average_precision_score(test_output, lr_predicter)
    #
    # print('Average precision-recall score: {0:0.2f}'.format(average_precision))


# # Divide dataset for training and testing
# training_inputs = input[:2000]
# training_outputs = label[:2000]
#
# testing_inputs = input[2000:]
# testing_outputs = label[2000:]



