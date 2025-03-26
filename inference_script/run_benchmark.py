import json
from pathlib import Path
import os

import numpy as np
from supervisely import VideoProject, OpenMode, VideoDataset

from src.benchmark.benchmark import (
    evaluate_frame_level,
    load_ground_truth,
    load_predictions,
)
from mouse_scripts.video_utils import get_total_frames


def get_figure(metrics):  # -> go.Figure:
    import plotly.graph_objects as go  # pylint: disable=import-error

    # Overall Metrics
    overall_metrics = metrics["overall"]
    r = [v for k, v in overall_metrics.items() if k != "support"]
    theta = [k for k, v in overall_metrics.items() if k != "support"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r + [r[0]],
            theta=theta + [theta[0]],
            # fill="toself",
            name="Overall Metrics",
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0.0, 1.0],
                ticks="outside",
            ),
            angularaxis=dict(rotation=90, direction="clockwise"),
        ),
        dragmode=False,
        margin=dict(l=25, r=25, t=25, b=25),
    )
    fig.update_layout(
        modebar=dict(
            remove=[
                "zoom2d",
                "pan2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ]
        )
    )
    return fig


def visuailize(benchmark_dir, metrics):
    from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget, SidebarWidget, ContainerWidget
    from supervisely.nn.benchmark.base_visualizer import BaseVisualizer
    from supervisely import Api
    
    class Visualizer(BaseVisualizer):
        def __init__(self, workdir, metrics):
            self.workdir = workdir
            self.metrics = metrics
            self.renderer = None
            self._widgets = None
            self._api = None
        
        @property
        def api(self):
            if self._api is None:
                print(os.getenv("SERVER_ADDRESS"))
                print(os.getenv("TEAM_ID"))
                self._api = Api()
            return self._api

        def _create_widgets(self):
            self.header = MarkdownWidget("header", "Header", text="# Benchmark Results")
            self.overview_chart = ChartWidget("key_metrics", get_figure(self.metrics))
            self._widgets = True
        
        
        def _create_layout(self):
            if not self._widgets:
                self._create_widgets()

            is_anchors_widgets = [
                # Overview
                (1, self.header),
                (1, self.overview_chart),
            ]

            anchors = []
            for is_anchor, widget in is_anchors_widgets:
                if is_anchor:
                    anchors.append(widget.id)

            sidebar = SidebarWidget(widgets=[i[1] for i in is_anchors_widgets], anchors=anchors)
            layout = ContainerWidget(
                widgets=[sidebar],
                name="main_container",
            )
            return layout

    vis_dir = str(benchmark_dir / Path("visualization"))
    visualizer = Visualizer(vis_dir, metrics=metrics)
    visualizer.visualize()
    print("Visualization saved to", vis_dir)
    try:
        team_id = int(os.environ["TEAM_ID"])
        visualizer.upload_results(team_id, "/test_benchmark/visualization")
    except Exception as e:
        print("Failed to upload visualization:", e)



if __name__ == "__main__":
    class_names = ["idle", "Self-Grooming", "Head/Body TWITCH"]
    conf = 0.6

    gt_path = "/gt"
    gt_dir_name = Path(os.environ.get('GT')).name
    project = VideoProject(gt_path, mode=OpenMode.READ)

    pred_path = "/pred"

    output_path = "/output"
    benchmark_dir = Path(output_path) / Path("benchmark")

    all_predictions = {}
    all_ground_truth = {}
    video_lengths = {}

    for dataset in project.datasets:
        dataset: VideoDataset
        for video_name, video_path, ann_path in dataset.items():
            predictions_path = Path(pred_path) / Path(dataset.path) / Path(f"{video_name}.json")
            if not predictions_path.exists():
                continue

            benchmark_results_path = Path(benchmark_dir) / Path(gt_dir_name) / Path(dataset.path) / Path(f"{video_name}.json")

            predictions = load_predictions(predictions_path, class_names, conf=conf)
            ground_truth = load_ground_truth(ann_path)
            
            print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth segments")

            # Evaluate frame-level metrics
            num_frames = get_total_frames(video_path)
            print(f"Total frames in video: {num_frames}")
            frame_level_results = evaluate_frame_level(predictions, ground_truth, num_frames, class_names[1:])
            print("\n=== Frame Level Evaluation ===")
            print(frame_level_results)
    
            data = {
                cls_name: {k:int(v) if isinstance(v, np.int64) else v for k, v in metrics.items()}
                for cls_name, metrics in frame_level_results.items()
            }

            os.makedirs(str(benchmark_results_path.parent), exist_ok=True)
            json.dump(data, open(benchmark_results_path, "w"), indent=4)

            print(f"\nMetrcis data for {video_name} saved to {benchmark_results_path}\n")

            # For aggregated metrics
            all_predictions[video_path] = predictions
            all_ground_truth[video_path] = ground_truth
            video_lengths[video_path] = num_frames

    # Evaluate frame-level metrics
    from src.benchmark.benchmark import evaluate_dataset_micro_average
    results = evaluate_dataset_micro_average(
        all_predictions,
        all_ground_truth,
        video_lengths,
        class_names,
    )
    results = {
        cls_name: {k:int(v) if isinstance(v, np.int64) else v for k, v in metrics.items()}
        for cls_name, metrics in results.items()
    }
    print("Evaluation Results:")
    print(results)
    
    # Save evaluation results
    evaluation_results_path = os.path.join(str(benchmark_dir), "aggregated_results.json")
    with open(evaluation_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Aggregated metrics saved to {benchmark_results_path}\n")

    visuailize(benchmark_dir, results)
