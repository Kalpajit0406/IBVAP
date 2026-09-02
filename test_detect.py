"""
Quick detection test — runs YOLO26n on an image or video file.

Usage:
    python test_detect.py path\to\image.jpg
    python test_detect.py path\to\video.mp4
    python test_detect.py https://example.com/photo.jpg   (URL, downloads first)

Output: annotated file saved next to the input, opened automatically.
"""
import sys
import urllib.request
from pathlib import Path

import cv2
from ultralytics import YOLO

CONF = 0.25
MODEL = "yolo26n.pt"


def detect_image(img_path: Path) -> None:
    model = YOLO(MODEL)
    result = model.predict(str(img_path), conf=CONF, verbose=False)[0]

    persons  = sum(1 for c in result.boxes.cls.tolist() if int(c) == 0)
    vehicles = sum(1 for c in result.boxes.cls.tolist() if int(c) in {2, 3, 5, 7})
    print(f"\n  Detections: {len(result.boxes)} total  |  persons={persons}  vehicles={vehicles}")
    for box in result.boxes:
        cid  = int(box.cls[0])
        conf = float(box.conf[0])
        name = result.names[cid]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"    {name:12s}  conf={conf:.2f}  bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

    out_path = img_path.parent / (img_path.stem + "_detected" + img_path.suffix)
    annotated = result.plot()
    cv2.imwrite(str(out_path), annotated)
    print(f"\n  Saved → {out_path}")

    # Try to open the image automatically
    try:
        import subprocess, os
        if sys.platform == "win32":
            os.startfile(str(out_path))
        else:
            subprocess.Popen(["xdg-open", str(out_path)])
    except Exception:
        pass


def detect_video(vid_path: Path) -> None:
    model  = YOLO(MODEL)
    cap    = cv2.VideoCapture(str(vid_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_p  = vid_path.parent / (vid_path.stem + "_detected.mp4")
    writer = cv2.VideoWriter(str(out_p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_n = 0
    print("  Processing video frames (Ctrl+C to stop early)…")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_n += 1
            result  = model.predict(frame, conf=CONF, verbose=False)[0]
            persons  = sum(1 for c in result.boxes.cls.tolist() if int(c) == 0)
            vehicles = sum(1 for c in result.boxes.cls.tolist() if int(c) in {2, 3, 5, 7})
            if persons or vehicles:
                print(f"    frame {frame_n:4d}  P={persons}  V={vehicles}")
            writer.write(result.plot())
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        writer.release()

    print(f"\n  Processed {frame_n} frames. Saved → {out_p}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_detect.py <image_or_video_path_or_url>")
        sys.exit(1)

    src = sys.argv[1]

    # Handle URL
    if src.startswith("http://") or src.startswith("https://"):
        suffix = Path(src).suffix or ".jpg"
        dest   = Path("test_input" + suffix)
        print(f"  Downloading {src} …")
        urllib.request.urlretrieve(src, dest)
        src = str(dest)

    path = Path(src)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"\nRunning YOLO26n (conf={CONF}) on: {path}")

    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        detect_video(path)
    else:
        detect_image(path)


if __name__ == "__main__":
    main()
