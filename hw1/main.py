# Author: Jiayi Liu

from math import log2

def read_file(file_name):
    try:
        f = open(file_name)
    except FileNotFoundError:
        print('The file does not exist')
        exit()
    else:
        # Init info list
        records = []     # training data records
        attributes = []  # all the names of attributes
        options = []     # corresponding options for each attribute
        # Read all lines from the file
        lines = f.readlines()
        num_lines = len(lines)
        # Extract all the attributes from the first line
        attributes = lines[0][1: -2].split(', ')[:-1]
        # Extract records
        for i in range(2, num_lines):
            records.append(lines[i][4:-2].split(', '))
        f.close()
        # Collect options for each attributes
        for i in range(len(attributes)):
            options.append(list(set([record[i] for record in records])))
        # Return the essential info for the decision trees
        return {"records": records, "attributes": attributes, "options": options}


class TreeNode:
    def __init__(self, value, index, trace, branch_from):
        self.value = value               # Name of attribute
        self.attr_index = index          # Index of attribute
        self.trace = trace               # Currently available attributes
        self.branch_from = branch_from   # Where does this node branch from (some option of the parent node's attribute)
        self.children = []               # List of children nodes
        self.branch_to = []              # List of options corresponding to each child

class DecisionTree:
    def __init__(self, info):
        self.records = info["records"]          # List of training data
        self.attributes = info["attributes"]    # List of attributes
        self.options = info["options"]          # List of possible options for each attribute
        self.dummy_root = TreeNode("dummy_root", -1, [attr for attr in range(len(self.attributes))], "null")
        self.test_data = []

    def train(self):
        self.build_trees(self.dummy_root, [record for record in range(len(self.records))], 'root')

    def predict(self, data):
        print('******** Prediction Begins ********')
        print('The traverse path is: ')
        self.test_data = data
        pred = self.traverse_tree(self.dummy_root.children[0])
        print('********* Prediction Ends *********')
        return pred

    def traverse_tree(self, node):
        result = ""
        print("--" + node.value)
        if node.attr_index >= 0:   # Split point
            option = self.test_data[node.attr_index]
            print(option)
            for i, branch in enumerate(node.branch_to):
                if option == branch:
                    result = self.traverse_tree(node.children[i])
        else:   # leaf
            return node.value
        return result

    def print_model(self):
        print("******** Model ********")
        self.print_tree(self.dummy_root.children[0])

    def print_tree(self, node, prefix="", last=True):
        print(prefix, "*- " if last else "|- ", node.value + "(" + node.branch_from + ")", sep="")
        prefix += "   " if last else "|  "
        child_count = len(node.children)

        for i, child in enumerate(node.children):
            last = i == (child_count - 1)
            self.print_tree(child, prefix, last)
        pass

    # The recursive function for building up the trees
    # Args:
    # parent_node: the parent node waited to be attached
    # data_pool: indices of records that collected to this branch
    def build_trees(self, parent_node, data_pool, branch_name):
        trace = parent_node.trace
        num_attr = len(trace)
        # Check termination condition: run out of attributes
        if num_attr == 0:
            # take majority as label
            leaf_node = self.get_majority(data_pool, branch_name)
            parent_node.children.append(leaf_node)
            return
        # Check termination condition: pure label
        if len(set([self.records[idx][-1] for idx in data_pool])) == 1:
            # assign unique label
            leaf_node = TreeNode(self.records[data_pool[0]][-1], -1, [], branch_name)
            parent_node.children.append(leaf_node)
            return

        # Compute IG for each available attribute
        IGs = []              # list of IG values for each attribute
        branches = []         # list of branches for each attribute
        branches_names = []   # list of option names for each branch
        # Compute the entropy before branch out
        entropy_before = self.entropy(data_pool)
        # Compute the information gain for each attribute
        for i in range(num_attr):
            IG, branches_pool, branches_name = self.information_gain(trace[i], data_pool, entropy_before)
            IGs.append(IG)
            branches.append(branches_pool)
            branches_names.append(branches_name)

        # Check termination condition: all the attributes are identical but with impure labels
        if len(set(IGs)) == 1:
            # take majority as label
            leaf_node = self.get_majority(data_pool, branch_name)
            parent_node.children.append(leaf_node)
            return

        # Select the attribute with the largest IG
        max_IG = 0
        max_index = 0  # max index of IGs
        for index, IG in enumerate(IGs):
            if IG > max_IG:
                max_IG = IG
                max_index = index

        # Update the trace for new node
        new_trace = trace.copy()
        new_trace.remove(trace[max_index])

        # Attach new node to the parent node
        new_node = TreeNode(self.attributes[trace[max_index]], trace[max_index], new_trace, branch_name)
        parent_node.children.append(new_node)

        # Recursion for each branch
        for branch_pool, name in zip(branches[max_index], branches_names[max_index]):
            new_node.branch_to.append(name)
            self.build_trees(new_node, branch_pool, name)
        pass

    # The function of computing information gain
    # Args:
    # attr_idx: the index of given attribute
    # data_pool: the indices of records
    # entropy_before: the entropy before branch out
    # Return:
    # IG: the value of information gain
    # branches_pool: a list of indices of the records for each branch
    def information_gain(self, attr_idx, data_pool, entropy_before):
        total = len(data_pool)
        branches_pool = []
        branches_name = []
        entropy_after = 0
        for option in self.options[attr_idx]:
            branch = [i for i in data_pool if self.records[i][attr_idx] == option]
            size_branch = len(branch)
            if size_branch > 0:
                prob = size_branch / total
                entropy_after += prob * self.entropy(branch)
                branches_pool.append(branch)
                branches_name.append(option)
        IG = entropy_before - entropy_after
        return IG, branches_pool, branches_name

    # The function of computing entropy for a given set of records
    # Args:
    # indices: the indices of records
    def entropy(self, indices):
        total = len(indices)
        cnt_yes = 0
        for index in indices:
            if self.records[index][-1] == 'Yes':
                cnt_yes += 1
        prob = cnt_yes / total
        if prob == 0 or prob == 1:
            return 0
        return prob * log2(1 / prob) + (1 - prob) * log2(1 / (1 - prob))

    # The function of taking the majority as the leaf node under the termination condition
    def get_majority(self, data_pool, branch_name):
        cnt_yes = len([True for idx in data_pool if self.records[idx][-1] == 'Yes'])
        cnt_no = len(data_pool) - cnt_yes
        leaf_node = TreeNode('Yes', -1, [], branch_name) if cnt_yes >= cnt_no else TreeNode('No', [], branch_name)
        return leaf_node


if __name__ == "__main__":
    # Input training data
    data = read_file('dt_data.txt')
    # Create a decision tree
    dt = DecisionTree(data)
    # Train the model
    dt.train()
    # Print out the model
    dt.print_model()
    # Test the model
    test_data = ['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']
    result = dt.predict(test_data)
    print("Prediction: " + result)
