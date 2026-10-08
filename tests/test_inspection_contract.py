"""Generated-image application acceptance; no dataset, model or scientific output."""

from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image

from visionguard.inspection_contract import (
    MAX_IMAGE_BYTES,
    InspectionError,
    ModelManifest,
    ModelOutput,
    prepare_image,
    validate_output,
)


def encoded(format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color=(20, 40, 60)).save(buffer, format=format)
    return buffer.getvalue()


def evidence(score=0.5):
    image = prepare_image(encoded())
    manifest = ModelManifest("manufactured-model", "a" * 64, "b" * 64, 0.5)
    output = ModelOutput(
        image.image_sha256,
        manifest.artifact_sha256,
        manifest.preprocessing_sha256,
        score,
        ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
    )
    return image, manifest, output


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_generated_input_identity_and_dimensions(format):
    data = encoded(format)
    image = prepare_image(data)
    assert (image.width, image.height) == (3, 2)
    assert image.image_bytes == data
    assert image == prepare_image(data)


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
@pytest.mark.parametrize("orientation", [0, 2, 3, 4, 5, 6, 7, 8, 9])
def test_exif_display_transform_refused_without_changing_model_pixels(
    format, orientation
):
    buffer = BytesIO()
    metadata = Image.Exif()
    metadata[274] = orientation
    Image.new("RGB", (3, 2), "red").save(buffer, format=format, exif=metadata)
    with pytest.raises(InspectionError, match="EXIF orientation"):
        prepare_image(buffer.getvalue())


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_upright_exif_keeps_original_coordinate_dimensions(format):
    buffer = BytesIO()
    metadata = Image.Exif()
    metadata[274] = 1
    Image.new("RGB", (3, 2), "red").save(buffer, format=format, exif=metadata)
    image = prepare_image(buffer.getvalue())
    assert (image.width, image.height) == (3, 2)


@pytest.mark.parametrize("payload", [b"", b"not an image", bytearray(b"x")])
def test_invalid_input_is_not_normal(payload):
    with pytest.raises(InspectionError):
        prepare_image(payload)


def test_oversize_encoded_input_rejected_before_decode():
    with pytest.raises(InspectionError, match="MiB"):
        prepare_image(b"x" * (MAX_IMAGE_BYTES + 1))


def test_decoded_pixel_limit(monkeypatch):
    monkeypatch.setattr("visionguard.inspection_contract.MAX_IMAGE_PIXELS", 5)
    with pytest.raises(InspectionError, match="pixel limit"):
        prepare_image(encoded())


def test_unsupported_format():
    with pytest.raises(InspectionError, match="PNG and JPEG"):
        prepare_image(encoded("GIF"))


def test_truncated_image():
    with pytest.raises(InspectionError):
        prepare_image(encoded()[:40])


def test_animated_input_refused():
    buffer = BytesIO()
    first = Image.new("RGB", (3, 2), "red")
    second = Image.new("RGB", (3, 2), "blue")
    first.save(buffer, format="PNG", save_all=True, append_images=[second])
    with pytest.raises(InspectionError, match="multi-frame"):
        prepare_image(buffer.getvalue())


def test_pillow_decompression_safeguard_remains_fail_closed(monkeypatch):
    data = encoded()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(InspectionError, match="decoding"):
        prepare_image(data)


@pytest.mark.parametrize(
    "field", ["model_id", "artifact_sha256", "preprocessing_sha256"]
)
def test_registry_paths_and_invalid_identities_refused(field):
    _, manifest, _ = evidence()
    with pytest.raises(InspectionError):
        replace(manifest, **{field: "../untrusted"})


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), True, "0.5"])
def test_invalid_calibration_refused(threshold):
    with pytest.raises(InspectionError):
        ModelManifest("manufactured-model", "a" * 64, "b" * 64, threshold)


@pytest.mark.parametrize(
    "field", ["image_sha256", "artifact_sha256", "preprocessing_sha256"]
)
def test_cross_request_or_model_result_refused(field):
    image, manifest, output = evidence()
    with pytest.raises(InspectionError, match="identities"):
        validate_output(image, manifest, replace(output, **{field: "c" * 64}))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_nonfinite_or_coerced_prediction_refused(score):
    image, manifest, output = evidence(score)
    with pytest.raises(InspectionError):
        validate_output(image, manifest, output)


@pytest.mark.parametrize(
    "map", [(), ((0.1,), (0.2,)), ((0.1, 0.2, 0.3), (0.4, float("nan"), 0.6))]
)
def test_invalid_map_refused(map):
    image, manifest, output = evidence()
    with pytest.raises(InspectionError):
        validate_output(image, manifest, replace(output, anomaly_map=map))


@pytest.mark.parametrize(
    "score, decision", [(0.49, "NORMAL"), (0.5, "NORMAL"), (0.51, "ANOMALOUS")]
)
def test_strict_preexisting_threshold_not_recalibrated(score, decision):
    image, manifest, output = evidence(score)
    assert validate_output(image, manifest, output) == decision
    assert manifest.threshold == 0.5
