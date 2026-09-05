"""
Multi-Camera Manager and Stream Synchronization Engine.
Manages concurrent camera connections, multi-threaded capture, fence boundary testing,
and timestamp-based cross-camera correlation.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import os
import cv2
import numpy as np

from utils.logger import get_logger
from utils.config_loader import load_config
from cameras.video_source import BaseVideoSource, VideoFile
from cameras.usb_camera import USBCamera
from cameras.rtsp_camera import RTSPCamera
from cameras.ip_camera import IPCamera
from cameras.drone_camera import DroneCamera
from cameras.image_folder import ImageFolder

logger = get_logger(__name__)


class CameraManager:
    """
    Coordinates active video sources, fence escape detection zones, and stream routing.
    """
    def __init__(self, config_path: str = "configs/cameras.yaml"):
        self.sources: Dict[str, BaseVideoSource] = {}
        self.fence_zones: Dict[str, List[List[int]]] = {}
        self.camera_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_cameras_config(config_path)

    def _load_cameras_config(self, path: str):
        if not os.path.exists(path):
            return

        import yaml
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}

            for cam in data.get("cameras", []):
                cid = cam.get("id")
                ctype = cam.get("type", "video_file")
                url = cam.get("url", "")
                name = cam.get("name", cid)

                source = self.create_source(ctype, cid, url, name=name, cfg=cam)
                if source:
                    self.sources[cid] = source
                    self.camera_metadata[cid] = cam

                    # Register virtual fence zone if defined
                    zone = cam.get("enclosure_zone", {})
                    poly = zone.get("polygon")
                    if poly and len(poly) >= 3:
                        self.fence_zones[cid] = poly
                        logger.info(f"Registered virtual fence zone '{zone.get('name', 'Zone')}' for {cid}")
        except Exception as e:
            logger.warning(f"Error loading cameras configuration from {path}: {e}")

    def create_source(
        self,
        source_type: str,
        source_id: str,
        uri_or_path: str,
        name: str = "",
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Optional[BaseVideoSource]:
        """Factory for building video source instances."""
        st = source_type.lower()
        cfg = cfg or {}

        if st == "usb":
            dev_idx = int(cfg.get("device_index", 0)) if uri_or_path == "" else int(uri_or_path)
            return USBCamera(
                source_id=source_id,
                device_index=dev_idx,
                width=cfg.get("resolution", [1280, 720])[0],
                height=cfg.get("resolution", [1280, 720])[1],
                fps=cfg.get("fps", 30.0),
            )
        elif st == "rtsp":
            return RTSPCamera(
                source_id=source_id,
                rtsp_url=uri_or_path,
                name=name or f"RTSP {source_id}",
                reconnect_delay_seconds=cfg.get("reconnect_delay_seconds", 5.0),
            )
        elif st == "ip_camera":
            return IPCamera(source_id=source_id, stream_url=uri_or_path, name=name)
        elif st == "drone":
            return DroneCamera(
                source_id=source_id,
                stream_url=uri_or_path,
                name=name or f"Drone {source_id}",
                telemetry_port=cfg.get("telemetry_port", 14550),
            )
        elif st == "image_folder":
            return ImageFolder(
                source_id=source_id,
                folder_path=uri_or_path,
                fps=cfg.get("fps", 5.0),
            )
        else:
            # Default to video file
            return VideoFile(
                source_id=source_id,
                file_path=uri_or_path,
                loop=cfg.get("loop", True),
            )

    def get_source(self, source_id: str) -> Optional[BaseVideoSource]:
        return self.sources.get(source_id)

    def register_source(self, source_id: str, source: BaseVideoSource):
        self.sources[source_id] = source

    def set_fence_zone(self, camera_id: str, polygon: List[List[int]]) -> None:
        """Sets or updates the virtual fence polygon for a camera."""
        if len(polygon) >= 3:
            self.fence_zones[camera_id] = polygon

    def is_inside_fence(self, camera_id: str, point: Tuple[float, float]) -> Tuple[bool, float]:
        """
        Tests whether an animal's centroid (x, y) is inside the virtual enclosure.
        Returns: (is_inside: bool, distance_to_boundary_px: float).
        Positive distance = inside, Negative = outside (escaped!).
        """
        poly = self.fence_zones.get(camera_id)
        if not poly or len(poly) < 3:
            # No fence configured -> considered safely inside
            return True, 100.0

        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        # cv2.pointPolygonTest: positive inside, 0 on edge, negative outside
        dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), measureDist=True)
        return (dist >= 0), float(dist)

    def correlate_detections_cross_camera(
        self,
        camera_detections: Dict[str, List[Dict[str, Any]]],
        time_window_sec: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Cross-camera correlation using timestamps, species, and visual heuristics.
        Allows tracks passing between overlapping camera fields of view to be linked.
        """
        correlated_groups = []
        # Multi-camera synchronization baseline
        for cam_a, dets_a in camera_detections.items():
            for det_a in dets_a:
                species = det_a.get("species")
                ts_a = det_a.get("timestamp", time.time())
                group = [det_a]

                for cam_b, dets_b in camera_detections.items():
                    if cam_a == cam_b:
                        continue
                    for det_b in dets_b:
                        if det_b.get("species") == species:
                            ts_b = det_b.get("timestamp", time.time())
                            if abs(ts_a - ts_b) <= time_window_sec:
                                group.append(det_b)

                if len(group) > 1:
                    correlated_groups.append({
                        "species": species,
                        "cameras": [d.get("camera_id") for d in group],
                        "detections": group,
                    })
        return correlated_groups

    def close_all(self):
        for s in self.sources.values():
            s.release()
        self.sources.clear()
