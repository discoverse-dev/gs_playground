"""MotrixSim Video Recorder for capturing rendering videos from batch environments.

This module provides a video recording utility for MotrixSim renderer that captures
only the first environment since MotrixSim doesn't support batch rendering.

Now supports multi-camera recording with independent video files per camera.
"""

from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Deque, Dict, Sequence
from collections import deque

import motrixsim as mtx
from motrixsim.render import RenderApp


class MotrixVideoRecorder:
    """Records MotrixSim rendering videos from batch environments.

    Since MotrixSim renderer doesn't support batch mode, this recorder only captures
    the first environment (env_id=0) from batch simulation data.

    Features:
        - Multi-camera support (cam_ids parameter)
        - Headless rendering (no interactive window)
        - Asynchronous camera capture with per-camera queue management
        - Automatic video encoding using OpenCV (one file per camera)
        - Backward compatible with single camera mode

    The recorder uses per-camera queues to manage pending capture tasks, since async
    captures may take multiple frames to complete. Each call to capture_frame() adds
    new tasks to all camera queues and checks if the oldest tasks are ready to write.
    """

    def __init__(
        self,
        model: mtx.SceneModel,
        output_path: str,
        fps: int = 30,
        img_width: int = 640,
        img_height: int = 480,
        max_pending_tasks: int = 10,
        cam_ids: Optional[Sequence[int]] = None,
    ):
        """Initialize the MotrixSim video recorder.

        Args:
            model: MotrixSim SceneModel for the environment
            output_path: Path to save output video(s). For multi-camera, used as base
                for generating camera-specific filenames.
            fps: Video frame rate (default: 30)
            img_width: Video width in pixels (default: 640)
            img_height: Video height in pixels (default: 480)
            max_pending_tasks: Maximum number of pending capture tasks per camera (default: 10)
            cam_ids: List of camera IDs to record (default: [0])

        Note:
            Video writers are not created until restart_episode() is called.
            This avoids creating temporary files before the actual episode starts.
        """
        # Normalize cam_ids (default to [0] for backward compatibility)
        if cam_ids is None:
            cam_ids = [0]
        self._cam_ids = [int(cid) for cid in cam_ids]

        # Validate camera IDs exist in model
        for cam_id in self._cam_ids:
            if cam_id >= len(model.cameras):
                raise ValueError(
                    f"Camera {cam_id} does not exist "
                    f"(model has {len(model.cameras)} cameras)"
                )

        # Generate output paths for each camera (will be regenerated in restart_episode)
        self._output_paths = self._generate_output_paths(output_path)

        # Set render targets for all cameras
        for cam_id in self._cam_ids:
            model.cameras[cam_id].set_render_target("image", img_width, img_height)

        # Create output directories if needed
        for path in self._output_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        # Shared resources
        self._model = model
        self._fps = fps
        self._img_width = img_width
        self._img_height = img_height
        self._max_pending_tasks = max_pending_tasks

        # Per-camera resources
        self._video_writers: Dict[int, cv2.VideoWriter] = {}
        self._pending_tasks: Dict[int, Deque] = {cid: deque() for cid in self._cam_ids}
        self._frames_captured: Dict[int, int] = {cid: 0 for cid in self._cam_ids}
        self._frames_written: Dict[int, int] = {cid: 0 for cid in self._cam_ids}

        # RenderApp (initialized in start())
        self._render: Optional[RenderApp] = None

    def _generate_output_paths(
        self, base_path: str, episode_index: Optional[int] = None
    ) -> Dict[int, Path]:
        """Generate output file paths for each camera.

        Rules:
            - Directory path: use episode naming if episode_index is provided, otherwise timestamped
            - Single camera file: use path as-is (backward compatible)
            - Multi-camera with file: use camera suffix pattern

        Args:
            base_path: Base output path (directory or file)
            episode_index: Optional episode index for generating episode-specific filenames

        Returns:
            Dictionary mapping camera ID to output file path
        """
        base = Path(base_path)

        # Directory path - generate episode files or timestamped files
        if base.is_dir() or base.suffix == "":
            if episode_index is not None:
                # Use episode index for filename
                if len(self._cam_ids) == 1:
                    # Single camera: simple filename
                    return {
                        self._cam_ids[0]: base
                        / f"motrix_episode_{episode_index:05d}.mp4"
                    }
                else:
                    # Multi-camera: include camera ID in filename
                    return {
                        cam_id: base
                        / f"motrix_episode_{episode_index:05d}_cam{cam_id}.mp4"
                        for cam_id in self._cam_ids
                    }
            else:
                # No episode index: use timestamp (for initial recording)
                import time

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if len(self._cam_ids) == 1:
                    # Single camera: simple filename
                    return {self._cam_ids[0]: base / f"motrix_render_{timestamp}.mp4"}
                else:
                    # Multi-camera: include camera ID in filename
                    return {
                        cam_id: base / f"motrix_render_cam{cam_id}_{timestamp}.mp4"
                        for cam_id in self._cam_ids
                    }

        # File path
        if len(self._cam_ids) == 1:
            # Single camera - use path as-is (backward compatible)
            return {self._cam_ids[0]: base}
        else:
            # Multi-camera with file pattern: insert camera suffix before extension
            stem = base.stem  # filename without extension
            suffix = base.suffix  # including dot (e.g., ".mp4")
            return {
                cam_id: base.parent / f"{stem}_cam{cam_id}{suffix}"
                for cam_id in self._cam_ids
            }

    def start(self):
        """Start the video recorder by initializing RenderApp.

        Video writers will be created when restart_episode() is called.
        This must be called before capture_frame() or restart_episode().
        """
        # Initialize headless RenderApp (shared across all cameras)
        self._render = RenderApp(log_level="WARN", headless=True)
        self._render.launch(self._model)

        cam_list = ", ".join(f"cam{cid}" for cid in self._cam_ids)
        print(f"[MotrixVideoRecorder] Started recording {cam_list}")

    def restart_episode(self, output_dir: str, episode_index: int):
        """Restart recording for a new episode.

        Closes current video writers and creates new ones with episode-specific filenames.

        Args:
            output_dir: Directory to save the new episode video
            episode_index: Episode index for generating filename
        """
        if len(self._video_writers) > 0:
            self._process_ready_tasks(block=True)
            # Close existing video writers
            for cam_id in self._cam_ids:
                writer = self._video_writers.get(cam_id)
                if writer is not None:
                    writer.release()

        # Generate new output paths with episode index
        self._output_paths = self._generate_output_paths(output_dir, episode_index)

        # Create new video writers
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for cam_id in self._cam_ids:
            path = self._output_paths[cam_id]
            writer = cv2.VideoWriter(
                str(path),
                fourcc,
                self._fps,
                (self._img_width, self._img_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {path}")
            self._video_writers[cam_id] = writer
            self._frames_captured[cam_id] = 0
            self._frames_written[cam_id] = 0
            self._pending_tasks[cam_id].clear()

        cam_list = ", ".join(f"cam{cid}" for cid in self._cam_ids)
        print(
            f"[MotrixVideoRecorder] Started new episode {episode_index} recording {cam_list}"
        )
        for cam_id in self._cam_ids:
            print(f"  cam{cam_id}: {self._output_paths[cam_id]}")

    def capture_frame(self, batch_data: mtx.SceneData):
        """Capture a frame from all configured cameras.

        Since camera capture is asynchronous and may take multiple frames to complete,
        this method:
        1. Syncs scene state once (shared across all cameras)
        2. Initiates async captures for all cameras
        3. Checks each camera's queue for ready tasks

        Args:
            batch_data: Batch SceneData from simulation (shape: [num_envs, ...])
        """
        if self._render is None:
            raise RuntimeError("MotrixVideoRecorder not started. Call start() first.")

        # Step 1: Extract first environment data and sync to renderer (once for all cameras)
        mask = np.zeros(batch_data.shape[0], dtype=bool)
        mask[0] = True
        single_data = batch_data[mask]

        # Step 2: Start async captures for all cameras
        for cam_id in self._cam_ids:
            camera = self._render.get_camera(cam_id)
            capture_task = camera.capture()
            self._pending_tasks[cam_id].append(capture_task)
            self._frames_captured[cam_id] += 1
        self._render.sync(single_data)
        # Step 3: Process ready tasks for all cameras
        self._process_ready_tasks()

    def _process_ready_tasks(self, block=False):
        """Process completed capture tasks for all cameras.

        Checks each camera's queue for ready tasks and writes them to video.
        Maintains FIFO ordering per camera.
        """
        for cam_id in self._cam_ids:
            queue = self._pending_tasks[cam_id]
            writer = self._video_writers[cam_id]
            if queue is None:
                continue
            while len(queue) > 0:
                if queue[0].state == "pending":
                    if not block:
                        break
                    else:
                        print(queue[0].state)
                        continue
                task: mtx.render.CaptureTask = queue.popleft()
                img = task.take_image()
                if img is not None:
                    pixels = img.pixels  # (H, W, 3) uint8
                    # MotrixSim returns RGB, OpenCV expects BGR
                    pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    writer.write(pixels_bgr)
                    self._frames_written[cam_id] += 1

    def stop(self):
        """Stop the recorder and finalize all video files.

        This must be called to properly save the videos. Waits for all pending
        capture tasks to complete before closing.
        """
        # Report pending frames
        total_pending = sum(len(q) for q in self._pending_tasks.values())
        if total_pending > 0:
            self._process_ready_tasks(block=True)

        # Close all video writers
        for cam_id in self._cam_ids:
            writer = self._video_writers.get(cam_id)
            if writer is not None:
                writer.release()
                print(
                    f"[MotrixVideoRecorder] Saved cam{cam_id}: {self._output_paths[cam_id]}"
                )
                print(
                    f"  Stats: {self._frames_written[cam_id]}/{self._frames_captured[cam_id]} frames"
                )

        self._video_writers.clear()

        # Close RenderApp
        if self._render is not None:
            self._render = None

    def __enter__(self):
        """Support with statement context manager."""
        self.start()
        return self

    def __exit__(self, *args):
        """Support with statement context manager for automatic cleanup."""
        self.stop()
