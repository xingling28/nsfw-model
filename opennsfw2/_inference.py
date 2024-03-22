"""
Inference utilities.
"""
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Sequence

import cv2
import numpy as np
from keras_core import Model
from PIL import Image  # type: ignore

from ._image import Preprocessing, preprocess_image
from ._typing import NDFloat32Array


def improved_predict_images(
    image_paths: Sequence[str],
    model: Model,
    batch_size: int = 8,
    preprocessing: Preprocessing = Preprocessing.YAHOO,
    grad_cam_paths: Optional[Sequence[str]] = None,
    alpha: float = 0.8,
) -> List[float]:
    """
    Pipeline from image paths to predicted NSFW probabilities.
    Optionally generate and save the Grad-CAM plots.
    """
    images = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            processed_image = preprocess_image(image, preprocessing)
            images.append(processed_image)

    images = np.array(images)

    predictions = model.predict(images, batch_size=batch_size, verbose=0)
    nsfw_probabilities: List[float] = predictions[:, 1].tolist()

    if grad_cam_paths is not None:
        # TensorFlow will only be imported here.
        from ._inspection import make_and_save_nsfw_grad_cam

        for image_path, grad_cam_path in zip(image_paths, grad_cam_paths):
            with Image.open(image_path) as image:
                make_and_save_nsfw_grad_cam(
                    image,
                    preprocessing,
                    model,
                    grad_cam_path,
                    alpha,
                )

    return nsfw_probabilities


class Aggregation(str, Enum):
    MEAN = auto()
    MEDIAN = auto()
    MAX = auto()
    MIN = auto()


def _get_aggregation_fn(
    aggregation: Aggregation,
) -> Callable[[NDFloat32Array], float]:
    def fn(x: NDFloat32Array) -> float:
        agg: Any = {
            Aggregation.MEAN: np.mean,
            Aggregation.MEDIAN: np.median,
            Aggregation.MAX: np.max,
            Aggregation.MIN: np.min,
        }[aggregation]
        return float(agg(x))

    return fn


def improved_predict_video_frames(
    video_path: str,
    model: Model,
    frames_to_check: list = None,
    clip_len: int = 10,
    preprocessing: Preprocessing = Preprocessing.YAHOO,
    batch_size: int = 5,
) -> float:
    cap = cv2.VideoCapture(video_path)
    value_to_return = 0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= clip_len:
            raise Exception(
                f"Video FPS ({fps}) must be greater than clip length ({clip_len}).",  # noqa
            )

        if not frames_to_check:
            np.random.seed(0)
            end_idx = np.random.randint(clip_len, fps)
            start_idx = end_idx - clip_len
            frames_to_check = np.linspace(start_idx, end_idx, num=clip_len)
            frames_to_check = np.clip(
                frames_to_check,
                start_idx,
                end_idx - 1,
            ).astype(np.int64)

        nsfw_probabilities = []
        input_frames_batch = []

        for frame_index in sorted(frames_to_check):
            while cap.get(cv2.CAP_PROP_POS_FRAMES) < frame_index:
                ret = cap.grab()
                if not ret:
                    break
            ret, bgr_frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            input_frame = preprocess_image(pil_frame, preprocessing)
            input_frames_batch.append(input_frame)

            if len(input_frames_batch) == batch_size:
                nsfw_probabilities.extend(
                    model(np.array(input_frames_batch))[:, 1],
                )
                input_frames_batch = []

        # Process remaining frames
        if input_frames_batch:
            nsfw_probabilities.extend(
                model(np.array(input_frames_batch))[:, 1],
            )

        value_to_return = (
            float(
                max(
                    nsfw_probabilities,
                )
            )
            if nsfw_probabilities
            else 0
        )
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        return value_to_return
