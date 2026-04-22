import pytz, os, json, re
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from microlane.schema.output import ModelPrediction


COLORS = ['#FFD700', '#00E5FF', '#FF4081', '#69FF47', '#FF6D00', '#E040FB']


class ExperimentEvaluate:

    def __init__(self, experiment_name) -> None:
        self.experiment_name = experiment_name
        self.file_name = "prediction.json"
        self.folder_dir = "results/testing/" + self.generate_folder_name() + "/inference"

    def store_prediction(self, prediction: ModelPrediction) -> None:
        if not os.path.exists(self.folder_dir):
            os.mkdir(self.folder_dir)

        end_file_path = self.folder_dir + "/" + self.file_name

        output_entry = {
            "lanes": prediction.lanes,
            "h_samples": prediction.sample.h_samples,
            "raw_file": prediction.sample.image_path,
            "run_time": prediction.run_time,
        }

        if os.path.exists(end_file_path):
            with open(end_file_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        else:
            data = []

        data.append(output_entry)

        with open(end_file_path, "w") as f:
            json.dump(data, f, indent=2)

    def visualize_prediction(
        self,
        prediction: ModelPrediction,
        show: bool = False,
    ) -> str:
        
        if not os.path.exists(self.folder_dir):
            os.mkdir(self.folder_dir)

        pattern = re.compile(r"visualization_(\d+)\.png")

        existing = [
            f for f in os.listdir(self.folder_dir)
            if f.startswith("visualization_") and f.endswith(".png")
        ]

        indices = [
            int(match.group(1))
            for f in existing
            if (match := pattern.search(f)) is not None
        ]

        viz_index = max(indices, default=-1) + 1

        save_path = os.path.join(
            self.folder_dir,
            f"visualization_{viz_index:04d}.png"
        )

        img = Image.open(prediction.sample.image_path)
        img_arr = np.array(img)

        modified_image = prediction.sample.image

        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        fig.suptitle(
            f"Inference time: {prediction.run_time:.4f}s  |  "
            f"File: {'/'.join(prediction.sample.image_path.split('/')[-3:])}",
            fontsize=11,
            color="gray",
        )

        axes[0].imshow(img_arr)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(modified_image)
        axes[1].set_title("Augmented", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(modified_image)
        axes[2].set_title("Predictions", fontsize=12)
        axes[2].axis("off")
        
        h, w = np.array(modified_image).shape[:2] # pyright: ignore[reportOptionalMemberAccess]
        axes[2].set_xlim(0, w)
        axes[2].set_ylim(h, 0)  # Note: inverted because image y-axis goes top-dow

        legend_patches = []

        for li, lane in enumerate(prediction.lanes):
            color = COLORS[li % len(COLORS)]
            xs, ys = [], []

            for x, y in zip(lane, prediction.sample.h_samples):
                if x == -2:
                    if xs:
                        # Bug fix: lane lines belong on axes[2], not axes[1]
                        axes[2].plot(xs, ys, color=color, linewidth=2)
                        xs, ys = [], []
                    elif 0 <= x < w and 0 <= y < h:  # only plot in-bounds points
                        xs.append(x)
                        ys.append(y)

            if xs:
                # Bug fix: flush the final segment to axes[2]
                axes[2].plot(xs, ys, color=color, linewidth=2)

            valid = [
                (x, y)
                for x, y in zip(lane, prediction.sample.h_samples)
                if x != -2
            ]
            if valid:
                vx, vy = zip(*valid)
                axes[2].scatter(vx, vy, color=color, s=10, zorder=5)

            legend_patches.append(mpatches.Patch(color=color, label=f"Lane {li + 1}"))

        axes[2].legend(
            handles=legend_patches,
            loc="upper left",
            fontsize=9,
            framealpha=0.6,
            facecolor="black",
            labelcolor="white",
        )

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        return save_path
    
    def generate_folder_name(self) -> str:
        experiment_name = self.experiment_name.lower().replace(" ", "_")
        timezone = pytz.timezone("Asia/Kathmandu")
        now = datetime.now(timezone)
        timestamp = now.strftime("%Y_%m_%d__%H_%M_%S")
        return f"{timestamp}_{experiment_name}"