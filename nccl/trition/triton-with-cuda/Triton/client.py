from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import tritonclient.grpc as grpcclient


colors = {}


def draw(
    image: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> None:
    """Draws the bounding boxes on the image.

    Args:
        image (np.ndarray): The input image (h, w, 3).
        bboxes (np.ndarray): The bounding boxes in (x1, y1, x2, y2) format.
        scores (np.ndarray): The scores of the bounding boxes.
        class_ids (np.ndarray): The class IDs of the bounding boxes.
    """
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        if class_id not in colors:
            colors[class_id] = np.random.randint(0, 256, size=3).tolist()
        color = colors[class_id]
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def run(client, image: np.ndarray):
    detection_input = grpcclient.InferInput("image", image.shape, "UINT8")
    detection_input.set_data_from_numpy(image)
    detection_response = client.infer("ensemble", inputs=[detection_input])
    bboxes = detection_response.as_numpy("bboxes")
    scores = detection_response.as_numpy("scores")
    class_ids = detection_response.as_numpy("ids")
    return bboxes, scores, class_ids


if __name__ == "__main__":
    # Setting up client
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    out_path = "/workspace/Results/triton.mp4"
    Path("/workspace/Results").mkdir(parents=True, exist_ok=True)
    verbose = False

    for _ in range(10):
        frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        bboxes, scores, class_ids = run(client, frame)

    # Load video
    cap = cv2.VideoCapture("/workspace/Assets/video.mp4")
    out = None

    t_infers = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        start = perf_counter()
        bboxes, scores, class_ids = run(client, frame)
        end = perf_counter()
        t_infers += end - start

        fps = 1 / (end - start)
        draw(frame, bboxes, scores, class_ids)
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        count += 1
        if verbose:
            print(f"{count} -> Inference: {(end - start) * 1e3:.3f} ms, FPS: {fps:.2f}")

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                20,
                (width, height),
            )
        out.write(frame)
    print(f"Inference: {t_infers * 1e3 / count :.3f} ms")
    print(f"FPS: {count / (t_infers):.2f}")
