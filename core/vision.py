"""Vision utilities: image encoding and video frame extraction for multimodal LLMs."""
from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

_log = logging.getLogger(__name__)


def encode_image_b64(path: Path) -> str:
    """Read an image file and return its base64-encoded content.

    Args:
        path: Path to the image file.

    Returns:
        Base64-encoded string of the image bytes.

    Raises:
        OSError: If the file cannot be read.
    """
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def extract_video_frames(path: Path, max_frames: int = 4) -> list[str]:
    """Extract evenly-spaced frames from a video and return them as base64 JPEG strings.

    Tries OpenCV first; falls back to ffmpeg if cv2 is not installed.

    Args:
        path: Path to the video file.
        max_frames: Maximum number of frames to extract (default 4).

    Returns:
        List of base64-encoded JPEG strings, one per extracted frame.
        Empty list if extraction fails.
    """
    frames = _extract_via_cv2(path, max_frames)
    if frames:
        return frames

    _log.info("cv2 not available — falling back to ffmpeg for %s", path.name)
    return _extract_via_ffmpeg(path, max_frames)


def _extract_via_cv2(path: Path, max_frames: int) -> list[str]:
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        return []

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        _log.warning("cv2: cannot open video %s", path)
        return []

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            _log.warning("cv2: cannot determine frame count for %s", path)
            return []

        # Evenly spaced indices across the video (avoid first/last 5% — often black)
        margin = max(1, int(total * 0.05))
        indices = [
            margin + int((total - 2 * margin) * i / max(max_frames - 1, 1))
            for i in range(max_frames)
        ]

        frames: list[str] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

        _log.info("cv2 extracted %d/%d frames from %s", len(frames), max_frames, path.name)
        return frames
    except Exception as exc:
        _log.warning("cv2 frame extraction failed for %s: %s", path.name, exc)
        return []
    finally:
        cap.release()


def _extract_via_ffmpeg(path: Path, max_frames: int) -> list[str]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_pattern = str(Path(tmp) / "frame%03d.jpg")
            # Extract 1 frame every N seconds to get roughly max_frames total
            # Use select filter to pick evenly-spaced frames
            result = subprocess.run(
                [
                    "ffmpeg", "-i", str(path),
                    "-vf", f"select=not(mod(n\\,{max(1, _estimate_frame_skip(path, max_frames))})),setpts=N/FRAME_RATE/TB",
                    "-frames:v", str(max_frames),
                    "-q:v", "3",
                    out_pattern,
                    "-y",
                ],
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                _log.warning("ffmpeg failed for %s: %s", path.name, result.stderr[:200])
                return []

            frame_files = sorted(Path(tmp).glob("frame*.jpg"))[:max_frames]
            frames = [
                base64.b64encode(f.read_bytes()).decode("utf-8")
                for f in frame_files
            ]
            _log.info("ffmpeg extracted %d frames from %s", len(frames), path.name)
            return frames
    except FileNotFoundError:
        _log.warning("ffmpeg not found — cannot extract video frames")
        return []
    except subprocess.TimeoutExpired:
        _log.warning("ffmpeg timed out on %s", path.name)
        return []
    except Exception as exc:
        _log.warning("ffmpeg extraction failed for %s: %s", path.name, exc)
        return []


def _estimate_frame_skip(path: Path, max_frames: int) -> int:
    """Estimate how many frames to skip between samples using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        total = int(result.stdout.strip())
        return max(1, total // max_frames)
    except Exception:
        return 30  # assume ~30 fps, 1 frame per second default
