# Imports
import pandas as pd
import numpy as np
import math

# Tree Class
class D_Tree():
    #Initializer
    def __init__(self, dataset, class_labels, attributes):
        # create the tree and save the root
        # Limit to max depth of three
        self.root = ID3_algorithm(dataset, dataset, class_labels, attributes, 3)
        
        # save the dataset
        self.data = dataset

    #Lets make a funciton to evaluate accuracy
    def get_accuracy(self, class_label):

        correct_count = 0
        #for each row in the dataset
        for index,row in self.data.iterrows():
            #print("Checking if " + str(row[class_label]) + " is equal to " + str(self.get_answer(row)))
            if(row[class_label] == self.get_answer(row)):
                #print("yes")
                correct_count += 1
        
        #print(correct_count)
        return correct_count/self.data[class_label].size

    # Given a single example(row), find the class label from travelling through the tree
    def get_answer(self, example):
        if self.root.isleaf:
            return self.root.label

        for node in self.root.below:
            if node.attribute in example.values.tolist():
                if(node.isleaf):
                    return node.label
                for sub_node in node.below:
                    if sub_node.attribute in example.values.tolist():
                        if(sub_node.isleaf):
                            return sub_node.label

    # For testing on unbinned data, and returning a [] of class labels
    def get_predictions(self, dataset):

        #array for storing predictions
        pred = []

        #for each row in the dataset
        for index,row in dataset.iterrows():
            #Assume False
            answer = 0          # Shouldn't really do this, but helps graphing over a larger range than the orginal dataset
            
            #Travel through the tree
            for node in self.root.below:
                
                #if the value of the dataset at the splitting attribute is in any interval/value contained in the nodes attribute
                if row[self.root.splitting_att] in node.attribute:

                    #get answer if leaf
                    if(node.isleaf):
                        answer = node.label
                    
                    #continue looking
                    for sub_node in node.below:
                        if row[node.splitting_att] in sub_node.attribute:
                            if(sub_node.isleaf):
                                answer = sub_node.label

            #print(f"{row} got this answer: {answer}")

            pred.append( answer )

        return pred

    # str override for tree
    def __str__(self):
        string = 'Root- ' + str(self.root)
        
        for child_node in self.root.below:
            string += '\n\tChild- ' + str(child_node)
            for grandchild in child_node.below:
                string += '\n\t\tGrandChild- ' + str(grandchild)

        return string


# Tree node class
class tree_node:
    #Initilize node attributes
    def __init__(self, data, isleaf, attribute):

        # 0 is a no , 1 is a yes
        self.label = -1

        self.below = [] # Stores all nodes below it

        self.data = data
        self.attribute = attribute  #attribute associated with node
        self.isleaf = isleaf
        self.splitting_att = None # attribute to split on when traversing the node
    
    # str override
    def __str__(self):
        return f"Tree Node: Label(0/1 yes/no) {self.label}|| Splitting: {self.splitting_att} Attribute: {self.attribute}"
    
    def __repr__(self):
        return str(self)


# Information Gain Calculation
def calculate_info_gain(column_label, class_label, dataset):

    # Find total entropy of dataset
    total_entropy = calculate_entropy(dataset[class_label])

    # For all possible bins ,  calculate entropy and sum

    subset_entropy_sum = 0
    counts = dataset[column_label].value_counts()

    #get all subset entropys
    for bin in pd.unique(dataset[column_label]):
        bin_entropy = calculate_entropy(dataset.loc[dataset[column_label] == bin][class_label])
        subset_entropy_sum += counts[bin] / dataset[column_label].size * bin_entropy

    #print(subset_entropy_sum)

    return total_entropy - subset_entropy_sum

# This function will find the best attribute to split on using the Information gain calculation
def best_attribute(dataset, class_label, attributes):

    highest_label = -1
    maxIG = -9999

    for att in attributes:
            IG = calculate_info_gain(att, class_label, dataset)
            if IG > maxIG:
                maxIG = IG
                highest_label = att
                #print(maxIG)
    #print(highest_label)

    return highest_label


# Tree building algorithm
# 
# Input: ORIGdataset unmodified dataset, dataset (pandas DataFrame... binned) , class_label, attributes (column labels)

def ID3_algorithm(ORIGdataset, dataset, class_label, attributes, depth):

    #Create a Root Node for the Tree
    root = tree_node(dataset, False, 'None')

    # Return the root if a couple different things are true
    unique_classes = pd.unique(dataset[class_label])

    if len(unique_classes) == 1:
        root.label = unique_classes[0]
        root.isleaf = True
        #print(root.label)
        return root

    if len(attributes) == 0 or depth == 1:
        root.label = dataset[class_label].value_counts().idxmax()
        root.isleaf = True
       # print(root.label)
        return root

    #Otherwise, begin

    #Find the best attribute to split on
    best_att = best_attribute(dataset, class_label, attributes)

    #Add to root
    root.splitting_att = best_att 
    

    for sub_attr in pd.unique(ORIGdataset[best_att]):
        # get the data for the sub attribute
        subset_data = dataset.loc[dataset[best_att] == sub_attr]


        if subset_data.empty:
            new_node = tree_node(dataset, True, sub_attr)
            new_node.label = dataset[class_label].value_counts().idxmax()# most common label within data
            #print(root.label)
            root.below.append( new_node )
        else:
            a_copy = attributes.copy()
            a_copy.remove(best_att)
            new_node = ID3_algorithm( ORIGdataset, subset_data, class_label, a_copy, depth -1)
            new_node.attribute = sub_attr
            root.below.append( new_node )
            root.label = -1

        #print(root.below)
    #end
    return root

# Calculates Entropy of a given class label array
# 
#   Input : 1-dimensional array of 1s and zeros, ex = [1,0,0,0,1,1,0,1,1,1,0,0]
def calculate_entropy(class_labels):

    #track the size for probabilities
    size = class_labels.size

    # Count the amount of 1s and zeros
    counts = [0,0]

    for x in class_labels:
        if(x == 0):
            counts[0] += 1
        else:
            counts[1] += 1

    # Calculate probabilities
    prob = [ (counts[0]/size) , (counts[1]/size) ]

    #Calculate Entropy and avoid log2(0)

    if prob[0] != 0:
        entropy_0 = -prob[0] * math.log2(prob[0])
    else:
        entropy_0 = 0
    
    if prob[1] != 0:
        entropy_1 = -prob[1] * math.log2(prob[1])
    else:
        entropy_1 = 0

    #return summation
    return entropy_0 + entropy_1 

