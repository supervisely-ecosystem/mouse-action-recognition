import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional, NamedTuple
from scipy.optimize import linear_sum_assignment
import numba
from collections import defaultdict


@dataclass(frozen=True)
class ActionSegment:
    """Class representing an action segment with temporal bounds and class label."""
    start_frame: int
    end_frame: int
    action_class: str
    confidence: float = 1.0  # Default confidence for ground truth, variable for predictions
    
    @property
    def duration(self) -> int:
        """Get the duration of the segment in frames."""
        return self.end_frame - self.start_frame + 1
    
    def __repr__(self) -> str:
        return f"ActionSegment({self.start_frame}-{self.end_frame}, {self.action_class}, {self.confidence:.2f})"


class Match(NamedTuple):
    """Represents a match between prediction and ground truth with its IoU score."""
    pred_idx: int
    gt_idx: int
    iou: float


@numba.njit
def temporal_iou(s1_start: int, s1_end: int, s2_start: int, s2_end: int) -> float:
    """
    Calculate temporal Intersection over Union (tIoU) between two segments.
    
    Args:
        s1_start: Start frame of first segment
        s1_end: End frame of first segment
        s2_start: Start frame of second segment
        s2_end: End frame of second segment
        
    Returns:
        float: tIoU value in range [0, 1]
    """
    # Calculate intersection bounds
    intersection_start = max(s1_start, s2_start)
    intersection_end = min(s1_end, s2_end)
    
    # Check if there is an actual intersection
    if intersection_end < intersection_start:
        return 0.0
    
    # Calculate areas
    intersection_area = intersection_end - intersection_start + 1
    s1_area = s1_end - s1_start + 1
    s2_area = s2_end - s2_start + 1
    union_area = s1_area + s2_area - intersection_area
    
    # Return IoU
    return float(intersection_area) / float(union_area)


class TemporalMatcher:
    """Efficient algorithm for matching predicted temporal segments with ground truth segments."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize the temporal matcher.
        
        Args:
            iou_threshold: Minimum IoU required for a valid match (default: 0.5)
        """
        self.iou_threshold = iou_threshold
    
    def match_segments(
        self, 
        predictions: List[ActionSegment], 
        ground_truth: List[ActionSegment],
        class_agnostic: bool = False
    ) -> Tuple[List[Match], List[int], List[int]]:
        """
        Match predicted segments to ground truth segments.
        
        Args:
            predictions: List of predicted ActionSegment objects
            ground_truth: List of ground truth ActionSegment objects
            class_agnostic: If True, ignore class labels during matching
            
        Returns:
            Tuple containing:
                - List of Match objects for matched segments
                - List of unmatched prediction indices
                - List of unmatched ground truth indices
        """
        # Handle empty cases
        if not predictions:
            return [], [], list(range(len(ground_truth)))
        if not ground_truth:
            return [], list(range(len(predictions))), []
        
        # Group segments by class for faster filtering (unless class_agnostic)
        gt_by_class = defaultdict(list)
        if not class_agnostic:
            for i, segment in enumerate(ground_truth):
                gt_by_class[segment.action_class].append(i)
        else:
            # If class_agnostic, put all GT in a single group
            for i in range(len(ground_truth)):
                gt_by_class["all"].append(i)
                
        # Pre-compute boundaries for faster interval tree construction
        pred_boundaries = np.array([(seg.start_frame, seg.end_frame) for seg in predictions])
        gt_boundaries = np.array([(seg.start_frame, seg.end_frame) for seg in ground_truth])
        
        # Find all potential matches
        potential_matches = self._find_potential_matches(
            predictions, ground_truth, 
            pred_boundaries, gt_boundaries,
            gt_by_class, class_agnostic
        )
        
        # If no valid matches, return all segments as unmatched
        if not potential_matches:
            return [], list(range(len(predictions))), list(range(len(ground_truth)))
        
        return self._greedy_matching(potential_matches, len(predictions), len(ground_truth))
    
    def _find_potential_matches(
        self,
        predictions: List[ActionSegment],
        ground_truth: List[ActionSegment],
        pred_boundaries: np.ndarray,
        gt_boundaries: np.ndarray,
        gt_by_class: Dict[str, List[int]],
        class_agnostic: bool
    ) -> List[Match]:
        """
        Find all potential matches between predictions and ground truth.
        
        Uses a sweep line algorithm for efficient overlap detection.
        """
        potential_matches = []
        
        # For each prediction, find potential ground truth matches
        for pred_idx, pred_segment in enumerate(predictions):
            # Get relevant ground truth indices to check
            relevant_gt_indices = gt_by_class["all"] if class_agnostic else gt_by_class.get(pred_segment.action_class, [])
            
            # Skip if no relevant ground truth
            if not relevant_gt_indices:
                continue
                
            # Use temporal bounds for fast filtering
            pred_start, pred_end = pred_boundaries[pred_idx]
            
            for gt_idx in relevant_gt_indices:
                gt_start, gt_end = gt_boundaries[gt_idx]
                
                # Skip if segments cannot overlap
                if gt_end < pred_start or gt_start > pred_end:
                    continue
                
                # Calculate IoU
                iou = temporal_iou(pred_start, pred_end, gt_start, gt_end)
                
                # Add as potential match if IoU exceeds threshold
                if iou >= self.iou_threshold:
                    potential_matches.append(Match(pred_idx, gt_idx, iou))
        
        # Sort by IoU in descending order
        return sorted(potential_matches, key=lambda m: m.iou, reverse=True)
    
    def _greedy_matching(
        self, 
        potential_matches: List[Match], 
        num_preds: int, 
        num_gt: int
    ) -> Tuple[List[Match], List[int], List[int]]:
        """
        Perform greedy matching by taking matches in order of decreasing IoU.
        """
        matched_pairs = []
        matched_preds = set()
        matched_gt = set()
        
        # Take matches in order of decreasing IoU
        for match in potential_matches:
            # Only consider this match if neither segment has been matched already
            if match.pred_idx not in matched_preds and match.gt_idx not in matched_gt:
                matched_pairs.append(match)
                matched_preds.add(match.pred_idx)
                matched_gt.add(match.gt_idx)
        
        # Determine unmatched segments
        unmatched_preds = [i for i in range(num_preds) if i not in matched_preds]
        unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
        
        return matched_pairs, unmatched_preds, unmatched_gt
    

class TemporalMetrics:
    """Calculate metrics for temporal action recognition."""
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Initialize with IoU thresholds for evaluation.
        
        Args:
            iou_thresholds: List of IoU thresholds to evaluate at (default: [0.5])
        """
        self.iou_thresholds = iou_thresholds or [0.5]
    
    def evaluate(
        self, 
        predictions: List[ActionSegment], 
        ground_truth: List[ActionSegment],
        class_wise: bool = True,
        class_agnostic: bool = False
    ) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of predicted ActionSegment objects
            ground_truth: List of ground truth ActionSegment objects
            class_wise: Whether to compute class-wise metrics
            class_agnostic: If True, ignore class labels during matching
            
        Returns:
            Dict containing overall and per-class metrics
        """
        results = {}
        
        # Sort predictions by confidence for AP calculation
        sorted_predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        
        # Calculate metrics at each IoU threshold
        for iou_threshold in self.iou_thresholds:
            matcher = TemporalMatcher(iou_threshold=iou_threshold)
            
            # Get matches
            matches, unmatched_preds, unmatched_gt = matcher.match_segments(
                sorted_predictions, ground_truth, class_agnostic
            )
            
            # Overall metrics
            num_predictions = len(predictions)
            num_ground_truth = len(ground_truth)
            num_matches = len(matches)
            
            precision = num_matches / num_predictions if num_predictions else 0
            recall = num_matches / num_ground_truth if num_ground_truth else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Average precision calculation
            ap = self._calculate_ap(sorted_predictions, ground_truth, iou_threshold, class_agnostic)
            
            results[f"AP@{iou_threshold}"] = ap
            results[f"Precision@{iou_threshold}"] = precision
            results[f"Recall@{iou_threshold}"] = recall
            results[f"F1@{iou_threshold}"] = f1
            
            # Class-wise metrics if requested
            if class_wise and not class_agnostic:
                results[f"class_metrics@{iou_threshold}"] = self._calculate_class_metrics(
                    sorted_predictions, ground_truth, iou_threshold
                )
        
        # Calculate mAP across thresholds
        ap_values = [results[f"AP@{t}"] for t in self.iou_thresholds]
        results["mAP"] = sum(ap_values) / len(ap_values) if ap_values else 0
        
        return results

    def _calculate_ap(
        self,
        predictions: List[ActionSegment],
        ground_truth: List[ActionSegment],
        iou_threshold: float,
        class_agnostic: bool = False
    ) -> float:
        """
        Calculate Average Precision for a specific IoU threshold.
        """
        if not predictions or not ground_truth:
            return 0.0
            
        # Create a copy of predictions sorted by confidence
        sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        
        # Initialize precision and recall arrays
        cumulative_tp = 0
        precision_values = []
        recall_values = []
        
        # Track which ground truth segments have been matched
        gt_matched = np.zeros(len(ground_truth), dtype=bool)
        
        # Process predictions in order of decreasing confidence
        for i, pred in enumerate(sorted_preds):
            # Check for match with any unmatched ground truth
            found_match = False
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                # Skip already matched ground truth
                if gt_matched[gt_idx]:
                    continue
                    
                # Skip different classes unless class_agnostic is True
                if not class_agnostic and pred.action_class != gt.action_class:
                    continue
                
                # Calculate IoU
                iou = temporal_iou(
                    pred.start_frame, pred.end_frame,
                    gt.start_frame, gt.end_frame
                )
                
                # Check if this is the best match so far
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    found_match = True
            
            # If match found, mark as true positive
            if found_match:
                cumulative_tp += 1
                gt_matched[best_gt_idx] = True
            
            # Calculate precision and recall at this point
            precision = cumulative_tp / (i + 1)
            recall = cumulative_tp / len(ground_truth)
            
            precision_values.append(precision)
            recall_values.append(recall)
        
        # Calculate AP using 101-point interpolation
        if not precision_values:
            return 0.0
            
        # Convert to numpy arrays
        precision_values = np.array(precision_values)
        recall_values = np.array(recall_values)
        
        # Add sentinel values for interpolation
        precision_values = np.concatenate([[0], precision_values, [0]])
        recall_values = np.concatenate([[0], recall_values, [1]])
        
        # Ensure precision is decreasing for interpolation
        for i in range(len(precision_values) - 1, 0, -1):
            precision_values[i-1] = max(precision_values[i-1], precision_values[i])
        
        # Find indices where recall changes
        indices = np.where(recall_values[1:] != recall_values[:-1])[0]
        
        # Sum up area under PR curve
        ap = np.sum((recall_values[indices + 1] - recall_values[indices]) * precision_values[indices + 1])
        
        return float(ap)
        
    def _calculate_class_metrics(
        self,
        predictions: List[ActionSegment],
        ground_truth: List[ActionSegment],
        iou_threshold: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.
        """
        # Get unique classes
        all_classes = set(seg.action_class for seg in predictions + ground_truth)
        
        class_metrics = {}
        
        for class_name in all_classes:
            # Filter by class
            class_preds = [p for p in predictions if p.action_class == class_name]
            class_gt = [g for g in ground_truth if g.action_class == class_name]
            
            # Skip if no ground truth for this class
            if not class_gt:
                class_metrics[class_name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "ap": 0.0,
                    "support": 0
                }
                continue
                
            # Match segments
            matcher = TemporalMatcher(iou_threshold=iou_threshold)
            matches, _, _ = matcher.match_segments(class_preds, class_gt)
            
            # Calculate metrics
            precision = len(matches) / len(class_preds) if class_preds else 0
            recall = len(matches) / len(class_gt) if class_gt else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate AP for this class
            ap = self._calculate_ap(class_preds, class_gt, iou_threshold)
            
            class_metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap": ap,
                "support": len(class_gt)
            }
        
        return class_metrics


# Example usage
if __name__ == "__main__":
    # Create sample data
    ground_truth = [
        ActionSegment(start_frame=10, end_frame=50, action_class="jump"),
        ActionSegment(start_frame=70, end_frame=100, action_class="run"),
        ActionSegment(start_frame=150, end_frame=200, action_class="jump"),
        ActionSegment(start_frame=240, end_frame=280, action_class="fall")
    ]
    
    predictions = [
        ActionSegment(start_frame=15, end_frame=55, action_class="jump", confidence=0.9),
        ActionSegment(start_frame=60, end_frame=105, action_class="run", confidence=0.8),
        ActionSegment(start_frame=155, end_frame=195, action_class="jump", confidence=0.7),
        ActionSegment(start_frame=230, end_frame=265, action_class="fall", confidence=0.6),
        ActionSegment(start_frame=300, end_frame=350, action_class="jump", confidence=0.5)  # False positive
    ]

    # from random import shuffle
    # shuffle(predictions)
    # shuffle(ground_truth)
    
    # Create matcher and find matches
    matcher = TemporalMatcher(iou_threshold=0.5)
    matches, unmatched_preds, unmatched_gt = matcher.match_segments(predictions, ground_truth)
    
    print("=== Matching Results ===")
    print(f"Found {len(matches)} matches at IoU threshold 0.5")
    for match in matches:
        pred = predictions[match.pred_idx]
        gt = ground_truth[match.gt_idx]
        print(f"Match: {pred} -> {gt} (IoU: {match.iou:.2f})")
    
    print(f"\nUnmatched predictions: {len(unmatched_preds)}")
    for idx in unmatched_preds:
        print(f"  {predictions[idx]}")
    
    print(f"\nUnmatched ground truth: {len(unmatched_gt)}")
    for idx in unmatched_gt:
        print(f"  {ground_truth[idx]}")
    
    # Calculate metrics
    metrics = TemporalMetrics(iou_thresholds=[0.3, 0.5, 0.7])
    results = metrics.evaluate(predictions, ground_truth)
    
    print("\n=== Metrics ===")
    print(f"mAP: {results['mAP']:.4f}")
    
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\nAt IoU={threshold}:")
        print(f"  AP: {results[f'AP@{threshold}']:.4f}")
        print(f"  Precision: {results[f'Precision@{threshold}']:.4f}")
        print(f"  Recall: {results[f'Recall@{threshold}']:.4f}")
        print(f"  F1: {results[f'F1@{threshold}']:.4f}")
    
    # Print class metrics at IoU=0.5
    print("\n=== Class Metrics (IoU=0.5) ===")
    class_metrics = results["class_metrics@0.5"]
    for class_name, metrics in class_metrics.items():
        print(f"\n{class_name.upper()}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")