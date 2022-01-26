"""
Ms. Namasivayam
ATCS 2021-2022
Binary Tree

Python program to for binary tree insertion and traversals
"""
from bst_node import Node


'''
A function that returns a string of the inorder 
traversal of a binary tree. 
Each node on the tree should be followed by a '-'.
Ex. "1-2-3-4-5-"
'''
def getInorder(root):

    if root.left != None:
        getInorder(root.left)

    print('' if root.val==None else f"{root.val}-", end="")

    if root.right != None:
       getInorder(root.right)



'''
A function that returns a string of the postorder 
traversal of a binary tree. 
Each node on the tree should be followed by a '-'.
Ex. "1-2-3-4-5-"
'''
def getPostorder(root):
    if root:
        getPostorder(root.left)
        getPostorder(root.right)
        print(f"{root.val}-",end="")


'''
A function that returns a string of the preorder 
traversal of a binary tree. 
Each node on the tree should be followed by a '-'.
Ex. "1-2-3-4-5-"
'''
def getPreorder(root):
    if root:
        print(f"{root.val}-", end="")
        getPreorder(root.left)
        getPreorder(root.right)


'''
A function that inserts a Node with the value
key in the proper position of the BST with the
provided root. The function will return the 
original root with no change if the key already
exists in the tree.
'''
def insert(root, key):
    if root == None:
        return Node(key)

    else:
        if root.val > key:
            root.left = insert(root.left, key)
            return root
        elif key > root.val:
            root.right = insert(root.right, key)
            return root
        elif key == root.val:
            return root


'''
Challenge: A function determines if a binary tree 
is a valid binary search tree
'''
def isBST(root):
    return False


if __name__ == '__main__':
    root = Node(10)
    root.left = Node(5)
    root.right = Node(15)
    root.left.left = Node(3)
    root.left.right = Node(9)

    print("Preorder traversal of binary tree is")
    getPreorder(root)

    print("\nInorder traversal of binary tree is")
    getInorder(root)

    print("\nPostorder traversal of binary tree is")
    getPostorder(root)

    root = insert(root, 8)
    print("\nInorder traversal of binary tree with 8 inserted is")
    getInorder(root)