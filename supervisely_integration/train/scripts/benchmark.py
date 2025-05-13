import os
from pathlib import Path

import supervisely.io.env as sly_env
from supervisely import logger
from inference_script.run_benchmark import (
    get_overview_chart_figure,
    get_per_video_table_data,
    get_per_class_table_data,
    get_overview_text,
    get_key_metrics_text,
)

def visualize(benchmark_dir, metrics, remote_dir, progress):
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