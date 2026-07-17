from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError


PathLike = Union[str, Path]


def image_to_rgb_array(image_path: PathLike) -> np.ndarray:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file does not exist: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"The provided path is not a file: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            rgb_array = np.asarray(rgb_image, dtype=np.uint8)

    except UnidentifiedImageError as error:
        raise ValueError(
            f"The file is not a readable image: {image_path}"
        ) from error

    except OSError as error:
        raise ValueError(
            f"The image could not be opened: {image_path}"
        ) from error

    return rgb_array