import json
import random
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

from xrkit.base import CONFIG
from xrkit.enhancement.filters import (
    bilateral_filter,
    contrast_limited_adaptative_histogram_equalization,
    dual_illumination_estimation,
    histogram_equalization,
    local_histogram_equalization,
    low_light_image_enhancement,
    total_variance_denoising,
)
from xrkit.enhancement.metrics.calculate import calculate_enhancement_metrics


class Enhancer:
    def __init__(self, n_samples: int, save_images: bool, generate_report: bool):
        self.save_images = save_images
        self.generate_report = generate_report

        self.technique_mapper = {
            "bilateral": bilateral_filter,
            "clahe": contrast_limited_adaptative_histogram_equalization,
            "dual": dual_illumination_estimation,
            "he": histogram_equalization,
            "lhe": local_histogram_equalization,
            "lime": low_light_image_enhancement,
            "tv": total_variance_denoising,
        }

        self.report_path = Path(CONFIG.reports.enhancement)
        self.data_path = Path(CONFIG.data.raw.path)
        self.save_path = Path(CONFIG.data.processed.path)

        image_suffix = CONFIG.base.image_suffix
        image_paths = [image for image in self.data_path.rglob(f"*.{image_suffix}")]

        random.seed(CONFIG.base.seed)
        self.image_paths = image_paths if not n_samples else random.sample(image_paths, n_samples)

    def run(self, technique: str):
        technique_function = self.technique_mapper.get(technique, None)
        if technique_function is None:
            raise ValueError("Invalid technique.")

        technique_save_path = Path(self.save_path, technique)
        technique_save_path.mkdir(parents=True, exist_ok=True)

        total_metrics: Dict[str, List[float]] = {}
        for path in tqdm(
            self.image_paths, desc=f"Enhancing with {technique_function.__name__} ({technique})"
        ):
            image = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
            result_image, execution_time = technique_function(image)

            if self.save_images:
                new_path = Path(technique_save_path, path.name)
                cv2.imwrite(new_path.as_posix(), result_image)

            current_metrics = calculate_enhancement_metrics(image, result_image)
            current_metrics["Execution Time"] = execution_time

            for metric, value in current_metrics.items():
                if metric not in total_metrics:
                    total_metrics[metric] = []

                total_metrics[metric].append(round(value, 3))

        if self.generate_report:
            technique_report_path = Path(self.report_path, technique).with_suffix(".json")
            with open(technique_report_path, "w") as file:
                json.dump(total_metrics, file, indent=4)
