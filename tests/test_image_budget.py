from PIL import Image

from edgemed_bench.run import resize_to_pixel_budget


def test_resize_to_pixel_budget_is_aspect_preserving_and_bounded() -> None:
    image = Image.new("RGB", (3124, 2401), "gray")
    resized = resize_to_pixel_budget(image, 786432)
    assert resized.width * resized.height <= 786432
    assert abs(resized.width / resized.height - image.width / image.height) < 0.01
    assert resize_to_pixel_budget(Image.new("RGB", (32, 32)), 786432).size == (32, 32)
