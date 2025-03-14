
def get_maximal_bbox(bboxes: list):
    """
    Get the maximal bounding box that encompasses all other boxes.
    Returns a bbox with min x1, min y1, max x2, max y2 from all bboxes.
    """
    if not bboxes:
        return None    
    min_x, min_y, max_x, max_y = bboxes[0]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    return min_x, min_y, max_x, max_y
