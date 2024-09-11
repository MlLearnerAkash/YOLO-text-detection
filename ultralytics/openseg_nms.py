import numpy as np

class Node:
    def __init__(self, bbox=None):
        self.bbox = bbox  # Bounding box coordinates
        self.children = []
        self.parent = None

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    union = area1 + area2 - intersection
    return intersection / union

def iol(box1, box2):
    """Calculate Intersection over Largest (IoL) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    return intersection / area2

def insert(bbox, node, n1, n2):
    """Insert a candidate bounding box into the tree structure."""
    for child in node.children:
        if iou(bbox, child.bbox) >= n1:
            return  # Exit if IoU is greater than or equal to N1
    
    for child in node.children:
        if iol(child.bbox, bbox) >= n2:
            insert(bbox, child, n1, n2)
            return
    
    for child in node.children:
        if iol(bbox, child.bbox) >= n2:
            child.parent = Node(bbox)
            node.children.remove(child)
            node.children.append(child.parent)
            child.parent.children.append(child)
            return
    
    new_node = Node(bbox)
    node.children.append(new_node)
    new_node.parent = node

def filter_bounding_boxes(bboxes, scores, n1, n2):
    """Filter bounding boxes using a tree structure based on IoU and IoL thresholds."""
    # Step 2: Sort bounding boxes by confidence scores
    sorted_indices = np.argsort(-np.array(scores))
    sorted_bboxes = [bboxes[i] for i in sorted_indices]
    
    # Step 3: Initialize the root node
    root = Node()
    
    # Step 12: Insert each bounding box into the tree
    for bbox in sorted_bboxes:
        insert(bbox, root, n1, n2)
    
    # Step 13: Collect all leaf nodes
    def collect_leaves(node):
        if not node.children:
            return [node.bbox]
        leaves = []
        for child in node.children:
            leaves.extend(collect_leaves(child))
        return leaves
    
    return collect_leaves(root)

# Example usage:
if __name__ == "__main__":
    bboxes = [
        [50, 50, 100, 100], 
        [55, 55, 105, 105], 
        [150, 150, 200, 200]
    ]
    scores = [1,1,1]
    n1 = 0.5  # IoU threshold
    n2 = 0.3  # IoL threshold

    filtered_boxes = filter_bounding_boxes(bboxes, scores, n1, n2)
    print("Filtered bounding boxes:", filtered_boxes)