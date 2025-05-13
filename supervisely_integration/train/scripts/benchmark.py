import os
from pathlib import Path

import supervisely.io.env as sly_env
from supervisely import logger


def get_overview_chart_figure(metrics):  # -> go.Figure:
    import plotly.graph_objects as go  # pylint: disable=import-error

    # Overall Metrics
    overall_metrics = metrics["aggregated"]["overall"]
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


def get_per_video_table_data(metrics: dict):
    metric_names = ["precision", "recall", "f1"]
    content = []
    for video_path, video_metrics in metrics.items():
        if video_path in ["aggregated"]:
            continue
        this_vid_metrics = []
        support = 0
        for metric_name in metric_names:
            value = video_metrics["weighted_avg"][metric_name]
            this_vid_metrics.append(
                f"{value:.4f}" if isinstance(value, float) else value
            )
        for class_name, class_metrics in video_metrics.items():
            if class_name not in ["weighted_avg", "macro_avg"]:
                support += class_metrics.get("support", 0)

        content.append({
            "id": video_path,
            "items": [video_path] + this_vid_metrics + [support],
        })
    table_data = {
        "columns": ["video_path", *metric_names, "support"],
        "content": content,
    }
    return table_data


def get_per_class_table_data(metrics: dict):
    metric_names = ["precision", "recall", "f1", "support"]
    content = [
        {
            "id": class_name,
            "items": [class_name] + [f"{class_metrics[metric_name]:.4f}" if isinstance(class_metrics[metric_name], float) else class_metrics[metric_name] for metric_name in metric_names],
        } for class_name, class_metrics in metrics["aggregated"].items() if class_name not in ["overall", "idle"]
    ]
    table_data = {
        "columns": ["Class", *metric_names],
        "content": content,
    }
    return table_data


def get_overview_text(metrics):
    class_names = [k for k in metrics["aggregated"].keys() if k not in ["overall", "idle"]]
    count = metrics["aggregated"]["overall"]["support"]
    text = f"""
- **Model:** MVD
- **Sample count:** {count}
- **Classes:** {', '.join(class_names)}
"""
    return text


def get_key_metrics_text():
    s = """
## Key Metrics:

- **Precision**: The proportion of correctly classified frames to the total number of frames predicted as positive. `(TP / (TP + FP))`
- **Recall**: The proportion of correctly classified frames to the total number of frames in that class. `(TP / (TP + FN))`
- **F1 Score**: The harmonic mean of precision and recall. `F1 = (2 * (Precision * Recall) / (Precision + Recall))`

Metrics are calculated based on the number of frames. The aggregation is done by **micro-averaging** (calculating metrics globally by counting the total true positives, false negatives and false positives for all classes, refer to the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support)).
"""
    return s

def visualize(benchmark_dir, remote_dir, metrics, progress):
    from supervisely import Api
    from supervisely.nn.benchmark.base_visualizer import BaseVisualizer
    from supervisely.nn.benchmark.visualization.widgets import (
        ChartWidget, ContainerWidget, MarkdownWidget, SidebarWidget,
        TableWidget)
    
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
                self._api = Api.from_env()
            return self._api

        def _create_widgets(self):
            self.header = MarkdownWidget("header", "Header", text="# Temporal Action Localization Metrics")
            self.overview = MarkdownWidget("overview", "Overview", text=get_overview_text(self.metrics))
            self.key_metrics_text = MarkdownWidget("key_metrics_text", "Key Metrics", text=get_key_metrics_text())
            self.key_metrics = ChartWidget("key_metrics", get_overview_chart_figure(self.metrics))
            self.per_video_text = MarkdownWidget("per_video_text", "Per Video", text="## Per Video Metrics")
            self.per_video_table = TableWidget("per_video_table", data=get_per_video_table_data(self.metrics))
            self.per_class_text = MarkdownWidget("per_class_text", "Per Class", text="## Per Class Metrics")
            self.per_class_table = TableWidget("per_class_table", data=get_per_class_table_data(self.metrics))
            self._widgets = True


        def _create_layout(self):
            if not self._widgets:
                self._create_widgets()

            is_anchors_widgets = [
                # Overview
                (0, self.header),
                (1, self.overview),
                (1, self.key_metrics_text),
                (0, self.key_metrics),
                (1, self.per_video_text),
                (0, self.per_video_table),
                (1, self.per_class_text),
                (0, self.per_class_table),
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
    logger.info(f"Visualization saved to: '{vis_dir}'")
    try:
        team_id = sly_env.team_id()
        remote_dir = os.path.join(remote_dir, "visualization")
        remote_dir = visualizer.upload_results(team_id, remote_dir, progress)
        logger.info(f"Visualization uploaded to Supervisely: '{remote_dir}' (team id: '{team_id}')")
        return remote_dir
    except Exception as e:
        logger.error("Failed to upload visualization:", e)