from entity_typing_framework.dataset_classes.datasets import BaseDataset
from entity_typing_framework.dataset_classes.KENN_datasets.kenn_utils import generate_constraints
from treelib import Tree

def create_tree(filepath, label2pred = False):
    # read data
    labels = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            labels.append(line.replace('\n',''))
    # sort labels
    labels.sort()
    # create tree
    tree = Tree()
    root = 'thing'
    tree.create_node(root,root)
    for label in labels:     
        # split levels
        splitted = label[1:].split('/')
        if len(splitted) == 1:
            # new first level node
            parent = root
        else:
            # init parent node
            parent = ''
            # convert to kenn predicate (must start with uppercase letter)
            if label2pred:
                parent = 'P'
            parent += '/' + '/'.join(splitted[:-1])
            if not tree.contains(parent):
                tree.create_node(parent,parent,root)
        # convert to kenn predicate (must start with uppercase letter)
        if label2pred:
            label = 'P' + label
        tree.create_node(label,label,parent)
    return tree


class KENNDataset(BaseDataset):

    def __init__(self, name, dataset_paths, types_file_path, clause_output_path, learnable_clause_weight, clause_weight, kb_mode):
        super().__init__(name = name, dataset_paths = dataset_paths)
        self.automatic_build_clauses(types_file_path, clause_output_path, learnable_clause_weight, clause_weight, kb_mode)

    def automatic_build_clauses(self, types_file_path, clause_output_path, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down'):
        ### CREATE KB ###
        # get ontology tree
        tree = create_tree(types_file_path, label2pred = True)
        # generate KENN clauses
        cw = '_' if learnable_clause_weight else clause_weight
        generate_constraints(tree, kb_mode, clause_output_path, cw)
