from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from src.benchmark.benchmark import ActionSegment


class SegmentVisualizer:
    """Class for visualizing action segments comparisons"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 5), style: str = 'default'):
        """
        Initialize the segment visualizer
        
        Args:
            figsize: Figure size (width, height) in inches
            style: Plot style ('default', 'dark', 'minimal')
        """
        self.figsize = figsize
        self.set_style(style)
        
    def set_style(self, style: str = 'default'):
        """Set the visual style of plots"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        if style == 'dark':
            plt.style.use('dark_background')
        elif style == 'minimal':
            plt.style.use('ggplot')
        else:
            plt.style.use('default')
    
    def visualize(
        self, 
        predictions: List[ActionSegment],
        ground_truth: List[ActionSegment],
        class_name: str,
        output_dir: str = 'results',
        title: Optional[str] = None,
        frame_range: Optional[Tuple[int, int]] = None
    ):
        """
        Visualize the comparison between predicted and ground truth segments
        
        Args:
            predictions: List of predicted ActionSegment objects
            ground_truth: List of ground truth ActionSegment objects
            class_name: Class name to visualize
            output_dir: Directory to save the output
            title: Custom title (if None, a default title is used)
            frame_range: Optional (start, end) tuple to limit the frame range
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter segments by class
        preds_filtered = [p for p in predictions if p.action_class == class_name]
        gt_filtered = [g for g in ground_truth if g.action_class == class_name]
        
        if not preds_filtered and not gt_filtered:
            print(f"No segments found for class '{class_name}'")
            return
        
        # Compute frame range 
        frame_range = self._compute_frame_range(preds_filtered, gt_filtered, frame_range)
        min_frame, max_frame = frame_range
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Define colors for visualization
        gt_color = '#1f77b4'    # Blue
        pred_color = '#ff7f0e'  # Orange
        
        # Draw segments using barh (horizontal bar) for cleaner visualization
        self._draw_segments(ax, gt_filtered, y_pos=0, color=gt_color, 
                           label='Ground Truth', frame_range=frame_range)
        self._draw_segments(ax, preds_filtered, y_pos=1, color=pred_color, 
                           label='Prediction', frame_range=frame_range)
        
        # Set up the axes
        self._setup_axes(ax, min_frame, max_frame, title or f"Action Segments for '{class_name}'")
        
        # Save the figure
        output_path = Path(output_dir) / f"segments_{class_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
        
        plt.close(fig)
    
    def _compute_frame_range(
        self, 
        predictions: List[ActionSegment], 
        ground_truth: List[ActionSegment],
        frame_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """Compute the frame range for visualization"""
        if frame_range is not None:
            return frame_range
            
        all_starts = ([p.start_frame for p in predictions] + 
                      [g.start_frame for g in ground_truth])
        all_ends = ([p.end_frame for p in predictions] + 
                   [g.end_frame for g in ground_truth])
        
        if all_starts and all_ends:
            min_frame = max(0, min(all_starts) - 50)
            max_frame = max(all_ends) + 50
        else:
            min_frame, max_frame = 0, 1000
            
        return min_frame, max_frame
    
    def _draw_segments(self, ax, segments, y_pos, color, label, frame_range):
        """Draw segments using horizontal bars for cleaner visualization"""
        min_frame, max_frame = frame_range
        
        # Filter segments that are within the frame range
        visible_segments = [s for s in segments 
                           if not (s.start_frame > max_frame or s.end_frame < min_frame)]
        
        if not visible_segments:
            return
            
        # Create a collection of bars for better performance
        lefts = [s.start_frame for s in visible_segments]
        widths = [s.duration for s in visible_segments]
        
        # Draw bars with a small height to look more like segments
        bar_height = 0.4
        bars = ax.barh(
            y=[y_pos] * len(visible_segments), 
            width=widths,
            left=lefts,
            height=bar_height,
            color=color,
            alpha=0.6,
            label=label,
            edgecolor=color,
            linewidth=1
        )
        
        # Add start/end markers for each segment
        for segment in visible_segments:
            ax.vlines(segment.start_frame, y_pos - bar_height/2, y_pos + bar_height/2, 
                     color=color, linewidth=2)
            ax.vlines(segment.end_frame, y_pos - bar_height/2, y_pos + bar_height/2, 
                     color=color, linewidth=2)
    
    def _setup_axes(self, ax, min_frame, max_frame, title):
        """Set up the axes with proper labels, ticks, grid, etc."""
        # Set axis limits and labels
        ax.set_xlim(min_frame, max_frame)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Ground Truth', 'Prediction'])
        ax.set_xlabel('Frame Number')
        
        # Add grid for better readability
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Set title
        ax.set_title(title)
        
        # Apply tight layout
        plt.tight_layout()


def draw_segments(
    predictions: List[ActionSegment], 
    ground_truth: List[ActionSegment], 
    class_name: str, 
    figsize: Tuple[int, int] = (15, 5),
    output_dir: str = 'results',
    title: Optional[str] = None,
    frame_range: Optional[Tuple[int, int]] = None
):
    """
    Draw a visual comparison between predicted and ground truth segments.
    
    Args:
        predictions: List of predicted ActionSegment objects
        ground_truth: List of ground truth ActionSegment objects
        class_name: Class name to visualize
        figsize: Figure size (width, height) in inches
        output_dir: Directory to save the output
        title: Custom title (if None, a default title is used)
        frame_range: Optional (start, end) tuple to limit the frame range
    """
    # Create a visualizer and use it
    visualizer = SegmentVisualizer(figsize=figsize)
    visualizer.visualize(
        predictions=predictions,
        ground_truth=ground_truth,
        class_name=class_name,
        output_dir=output_dir,
        title=title,
        frame_range=frame_range
    )