import os
import pandas
import numpy
import csv
import random
import copy
import sys


L = sys.argv[1]
K = sys.argv[2]
training_set = sys.argv[3]
validation_set = sys.argv[4]
test_set = sys.argv[5]
to_print = sys.argv[6]

cwd = os.getcwd()

data = csv.reader(open(cwd + training_set))
data_frame = []
for row in data:
    data_frame.append(row)
attributes = data_frame[0]
data_frame.pop(0)
headers = []

for i in range(len(attributes)):
    headers.append(attributes[i])

dataFrame = pandas.DataFrame(data_frame, columns=attributes)

target_classifier = []
target_classifier = dataFrame.Class
headers.remove('Class')

validate_data = csv.reader(open(cwd + validation_set))
validate_data_frame = []
for row in validate_data:
    validate_data_frame.append(row)
attributes = validate_data_frame[0]
validate_data_frame.pop(0)
# headers = []
# for i in range(len(attributes)):
#     headers.append(attributes[i])
validate_dataFrame = pandas.DataFrame(validate_data_frame, columns=attributes)

test_data = csv.reader(open(cwd + test_set))
test_data_frame = []
for row in test_data:
    test_data_frame.append(row)
attributes = test_data_frame[0]
test_data_frame.pop(0)
# headers = []
# for i in range(len(attributes)):
#     headers.append(attributes[i])
test_dataFrame = pandas.DataFrame(test_data_frame, columns=attributes)


# headers.remove('Class')



def calculate_Entropy(target_classifier):
    _, val_freqs = numpy.unique(target_classifier, return_counts=True)
    entropy = 0
    for i in val_freqs:
        entropy += (- (i / len(target_classifier) * numpy.log2(i / len(target_classifier))))
    return entropy


def calculate_gain(attr, t_classifier):
    index0 = [i for i, j in enumerate(attr) if j == '0']
    index1 = [i for i, j in enumerate(attr) if j == '1']
    target_classifier_0 = [t_classifier[i] for i in index0]
    target_classifier_1 = [t_classifier[i] for i in index1]
    entropy_classifier_0 = (len(target_classifier_0) / len(t_classifier)) * calculate_Entropy(target_classifier_0)
    entropy_classifier_1 = (len(target_classifier_1) / len(t_classifier)) * calculate_Entropy(target_classifier_1)
    info_gain = calculate_Entropy(t_classifier) - entropy_classifier_0 - entropy_classifier_1;
    return info_gain;


def calculate_varianceImpurity(target_classifier):
    _, val_freqs = numpy.unique(target_classifier, return_counts=True)
    variance_impurity = 1
    for i in val_freqs:
        variance_impurity *= (-(float(i) / len(target_classifier)))

    return variance_impurity


def calculate_varianceGain(attr, t_classifier):
    index0 = [i for i, j in enumerate(attr) if j == '0']
    index1 = [i for i, j in enumerate(attr) if j == '1']
    target_classifier_0 = [t_classifier[i] for i in index0]
    target_classifier_1 = [t_classifier[i] for i in index1]
    varImpurity_classifier_0 = (float(len(target_classifier_0)) / len(t_classifier)) * calculate_varianceImpurity(
        target_classifier_0)
    varImpurity_classifier_1 = (float(len(target_classifier_1)) / len(t_classifier)) * calculate_varianceImpurity(
        target_classifier_1)
    varImpurity_gain = calculate_varianceImpurity(t_classifier) - varImpurity_classifier_0 - varImpurity_classifier_1;
    # print(varImpurity_gain)
    return varImpurity_gain;


def chooseBestAttr(data_frame, headers, t_classifier, entropy_Method):
    best = headers[0]
    maxGain = 0
    if (entropy_Method):
        for attr in headers:
            newGain = calculate_gain(data_frame[attr], t_classifier)
            if newGain > maxGain:
                maxGain = newGain
                best = attr

    else:
        for attr in headers:
            newGain = calculate_varianceGain(data_frame[attr], t_classifier)
            if newGain > maxGain:
                maxGain = newGain
                best = attr

    return best


class Tree_Node(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        self.majority = None


def ID3(data_frame, t_classifier, headerList, AttrChoice):
    root = Tree_Node()
    values, val_freqs = numpy.unique(t_classifier, return_counts=True)
    maxValue = values[numpy.argmax(val_freqs)]
    # print(val_freqs)
    if (len(values) == 1):
        if (values[0] == '0'):
            # if(values[0] is not None and values[0] == '0'):
            root.data = '0'
            root.majority = '0'
            return root
        elif (values[0] == '1'):
            root.data = '1'
            root.majority = '1'
            return root

    if (len(headerList) == 0):
        _, val_freqs = numpy.unique(t_classifier, return_counts=True)
        if (val_freqs[0] > val_freqs[1]):
            root.data = '0'
            root.majority = '0'
        else:
            root.data = '1'
            root.majority = '1'
        return root
    else:
        bestAttr = chooseBestAttr(data_frame, headerList, t_classifier, AttrChoice)

        root.data = bestAttr

        root.majority = maxValue

        headerListCopy = copy.copy(headerList)

        headerListCopy.remove(bestAttr)


        root_child = Tree_Node()

        index0 = [i for i, j in enumerate(data_frame[bestAttr]) if j == '0']
        target_classifier_0 = [t_classifier[i] for i in index0]
        data_frame_0 = [data_frame.iloc[i] for i in index0]
        data_frame_0 = pandas.DataFrame(data_frame_0)
        if (len(data_frame_0) == 0):

            if (val_freqs[0] > val_freqs[1]):
                root_child.data = '0'
                root_child.majority = '0'
                root.left = root_child
            else:
                root_child.data = '1'
                root_child.majority = '1'
                root.left = root_child
        else:
            root.left = ID3(data_frame_0, target_classifier_0, headerListCopy, AttrChoice)

        index1 = [i for i, j in enumerate(data_frame[bestAttr]) if j == '1']
        target_classifier_1 = [t_classifier[i] for i in index1]
        data_frame_1 = [data_frame.iloc[i] for i in index1]
        data_frame_1 = pandas.DataFrame(data_frame_1)

        if (len(data_frame_1) == 0):

            if (val_freqs[0] > val_freqs[1]):
                root_child.data = '0'
                root_child.majority = '0'
                root.right = root_child
            else:
                root_child.data = '1'
                root_child.majority = '1'
                root.right = root_child
        else:
            root.right = ID3(data_frame_1, target_classifier_1, headerListCopy, AttrChoice)

        return root


def traverse(rootnode):
    treeNodeList = []
    thislevel = [rootnode]
    while thislevel:
        nextlevel = list()
        for n in thislevel:
            # print (n.data),
            if (n.data is not '0' and n.data is not '1'):
                treeNodeList.append(n)
            if n.left: nextlevel.append(n.left)
            if n.right: nextlevel.append(n.right)
        # print()
        thislevel = nextlevel
    return treeNodeList


def TREE_PRINT(tree, indent=''):
    if tree.data is not None:
        str = tree.data
    if tree.left is not None and tree.right is not None:
        data_str_0 = str + ' = 0 : '
        if checkIfLeafNode(tree.left):
            data_str_0 = data_str_0 + tree.left.data
        print(indent, data_str_0)
        TREE_PRINT(tree.left, indent + ' | ')
        data_str_1 = str + ' = 1 : '
        if checkIfLeafNode(tree.right):
            data_str_1 = data_str_1 + tree.right.data
        print(indent, data_str_1)
        TREE_PRINT(tree.right, indent + ' | ')


def checkIfLeafNode(tree):
    if tree.left is None and tree.right is None:
        return True
    else:
        return False


def validate_set(validate_dataFrame_record, root):
    while (root.data is not '0' and root.data is not '1'):
        if validate_dataFrame_record[root.data] == '0':
            root = root.left;
        elif validate_dataFrame_record[root.data] == '1':
            root = root.right;
    return '0' if root.data is '0' else '1'


def tree_copy(newroot, oldroot):
    if oldroot.data is not None:
        newroot.data = oldroot.data
        newroot.majority = oldroot.majority

    if oldroot.left is not None:
        newleftchild = Tree_Node()
        newroot.left = tree_copy(newleftchild, oldroot.left)
    if oldroot.right is not None:
        newrightchild = Tree_Node()
        newroot.right = tree_copy(newrightchild, oldroot.right)
    return newroot


def replace_subtree(prunedroot, root, attr, target_value):
    if root.data is not None and root.data is not attr:
        prunedroot.data = root.data
        prunedroot.majority = root.majority
    elif root.data is attr and root.data is not None:
        prunedroot.data = target_value
        prunedroot.majority = target_value
        return prunedroot
    if root.left is not None:
        newleftchild = Tree_Node()
        prunedroot.left = replace_subtree(newleftchild, root.left, attr, target_value)
    if root.right is not None:
        newrightchild = Tree_Node()
        prunedroot.right = replace_subtree(newrightchild, root.right, attr, target_value)
    return prunedroot


def accuracy(class_dt, class_df):
    correctSamples = [1 for a, b in zip(class_df, class_dt) if a == b]
    return len(correctSamples) / len(class_dt)


def postPruning(L, K):
    headerList = copy.copy(headers)
    N = 1
    while (N <= 2):

        if (N == 1):
            root = ID3(dataFrame, target_classifier, headerList, True)
            print("TREE with Entropy")
        else:
            root = ID3(dataFrame, target_classifier, headerList, False)
            print("TREE with Variance")
        N = N + 1

        if to_print == 'yes':
            # print("ID3 Tree")
            TREE_PRINT(root)

        bestTree = tree_copy(Tree_Node(), root)
        validate_class_org = [validate_set(validate_dataFrame.iloc[i], root) for i in range(len(validate_dataFrame))]
        validate_org_accuracy = accuracy(validate_class_org, validate_dataFrame.Class)
        print('Original Accuracy : validate set', validate_org_accuracy)
        test_class_org = [validate_set(test_dataFrame.iloc[i], root) for i in range(len(test_dataFrame))]
        test_org_accuracy = accuracy(test_class_org, test_dataFrame.Class)
        print('Original Accuracy : test set', test_org_accuracy)

        newroot = Tree_Node()
        for i in range(L):
            newroot = tree_copy(newroot, bestTree)
            m = random.randint(0, K)
            for j in range(m):
                treeNodeList = traverse(root)
                p = random.randint(0, len(treeNodeList) - 1)
                leafNode = Tree_Node()
                leafNode.data = treeNodeList[p].majority
                leafNode.majority = treeNodeList[p].majority

                prunedTree = replace_subtree(Tree_Node(), root, treeNodeList[p], leafNode)
                if to_print == 'yes':
                    # print("ID3 Tree")
                    TREE_PRINT(prunedTree)
            validate_class_pruned = [validate_set(validate_dataFrame.iloc[i], bestTree) for i in
                                     range(len(validate_dataFrame))]
            validate_pruned_accuracy = accuracy(validate_class_pruned, validate_dataFrame.Class)
            if validate_org_accuracy < validate_pruned_accuracy:
                bestTree = tree_copy(bestTree, prunedTree)

            validate_class_pruned = [validate_set(validate_dataFrame.iloc[i], bestTree) for i in
                                     range(len(validate_dataFrame))]
            validate_pruned_accuracy = accuracy(validate_class_pruned, validate_dataFrame.Class)
            test_class_pruned = [validate_set(test_dataFrame.iloc[i], bestTree) for i in range(len(test_dataFrame))]
            test_pruned_accuracy = accuracy(test_class_pruned, test_dataFrame.Class)

            print("Accuracy after pruning at", i, " : validate set", validate_pruned_accuracy)
            print("Accuracy after pruning at", i, ": test set", test_pruned_accuracy)

    return bestTree


root_tree = Tree_Node()
postPruning(int(L), int(K))
# root = ID3(dataFrame,target_classifier,headers,False)
