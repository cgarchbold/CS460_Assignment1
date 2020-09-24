import pandas as pd
import math

class D_Tree():
    def __init__(self, dataset, class_labels, attributes):
        self.root = ID3_algorithm(dataset, class_labels, attributes)

    # Lets make a simple recursive function to travel down the array
    # Lets assume input is in the form ['x','y', 'z'
    def get_answer(self,example , tree_node):
        if(tree_node.isleaf):
            return tree_node.label
        for child_node in self.root.below:
            if child_node.attribute in example:
                return self.get_answer(example, child_node)

    def __str__(self):
        string = 'Root- ' + str(self.root)
        
        for child_node in self.root.below:
            string += '\n\tChild- ' + str(child_node)
            for grandchild in child_node.below:
                string += '\n\t\tGrandChild- ' + str(grandchild)

        return string



class tree_node:
    def __init__(self, data, isleaf, attribute):

        # 0 is a no , 1 is a yes
        self.label = -1
        self.below = []
        self.data = data
        self.attribute = attribute
        self.isleaf = isleaf
    
    def __str__(self):
        return f"Tree Node: Label(0/1 yes/no) {self.label} Attribute: {self.attribute}"
    
    def __repr__(self):
        return str(self)


# Information Gain Calculation
def calculate_info_gain(column_label, class_label, dataset):

    # Find total entropy of dataset
    total_entropy = calculate_entropy(dataset[class_label])

    # For all possible bins ,  calculate entropy and sum

    subset_entropy_sum = 0
    counts = dataset[column_label].value_counts()

    for bin in pd.unique(dataset[column_label]):
        bin_entropy = calculate_entropy(dataset.loc[dataset[column_label] == bin][class_label])
        subset_entropy_sum += counts[bin] / dataset.size * bin_entropy

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

    return highest_label


# Tree building algorithm
# 
# Input: dataset (pandas DataFrame) , class_label, attributes (column labels)

def ID3_algorithm(dataset, class_label, attributes):

    #Create a Root Node for the Tree
    root = tree_node(dataset, False, 'None')

    # Return the root if a couple different things are true
    unique_classes = pd.unique(dataset[class_label])

    if len(unique_classes) == 1:
        root.label = unique_classes[0]
        return root

    if len(attributes) == 0:
        root.label = dataset[class_label].value_counts().idxmax()
        return root

    #Otherwise, begin

    #Find the best attribute to split on
    best_att = best_attribute(dataset, class_label, attributes)

    #Add to root
    root.attribute = best_att 
    

    for sub_attr in pd.unique(dataset[best_att]):
        # get the data for the sub attribute
        subset_data = dataset.loc[dataset[best_att] == sub_attr]

        #add a new tree branch below the root
        #new_node = tree_node(data, False, sub_attr)# sub set of the data set for val)
        #new_node.attribute = sub_attr a

        
        #print(dataset.loc[dataset[best_att] == val])

        if subset_data.empty:
            new_node = tree_node(dataset, True, sub_attr)
            new_node.label = dataset[class_label].value_counts().idxmax()# most common label within data
            root.below.append( new_node )
        else:
            a_copy = attributes.copy()
            a_copy.remove(best_att)
            new_node = ID3_algorithm(subset_data, class_label, a_copy)
            new_node.attribute = sub_attr
            root.below.append( new_node )

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

