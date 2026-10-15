"""Portable, versioned scene and recording archives. Never extract archives to disk."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

FORMAT_VERSION = 1
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def import_spec(path: str) -> mujoco.MjSpec:
    """Resolve includes and embed mesh/texture data, so projects survive relocation."""
    source = Path(path).expanduser().resolve()
    spec = mujoco.MjSpec.from_file(str(source))
    spec.compile()  # validate includes and resources before replacing the active world
    root = ET.fromstring(spec.to_xml())
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    assets = {}
    for asset in root.findall("./asset/*"):
        for key, value in list(asset.attrib.items()):
            if not key.startswith("file") or not value:
                continue
            directory = meshdir if asset.tag == "mesh" else texturedir if asset.tag == "texture" else ""
            location = source.parent / directory / value
            payload = spec.assets.get(value)
            if payload is None:
                payload = location.read_bytes()
            filename = f"resource_{len(assets):04d}{Path(value).suffix}"
            assets[filename] = payload
            asset.set(key, filename)
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    portable = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"), assets=assets)
    portable.compile()
    return portable


def save_archive(path: str, spec: mujoco.MjSpec, metadata: dict, states=None, times=None):
    destination = Path(path).expanduser().resolve()
    if not destination.parent.is_dir():
        raise ValueError("保存目录不存在")
    metadata = {**metadata, "format": "mjstudio", "version": FORMAT_VERSION,
                "mujoco_version": mujoco.__version__}
    fd, temporary = tempfile.mkstemp(prefix=".mjstudio-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            with zipfile.ZipFile(file, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(metadata, ensure_ascii=False, allow_nan=False))
                archive.writestr("scene.xml", spec.to_xml())
                for name, content in spec.assets.items():
                    archive.writestr("assets/" + name, content)
                if states is not None:
                    for name, values in (("states.npy", states), ("times.npy", times)):
                        buffer = io.BytesIO()
                        np.save(buffer, np.asarray(values), allow_pickle=False)
                        archive.writestr(name, buffer.getvalue())
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_archive(path: str):
    with zipfile.ZipFile(Path(path).expanduser(), "r") as archive:
        if sum(entry.file_size for entry in archive.infolist()) > MAX_ARCHIVE_BYTES:
            raise ValueError("项目解压后超过 512 MB 限制")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "mjstudio" or manifest.get("version") != FORMAT_VERSION:
            raise ValueError("不支持的项目格式或版本")
        assets = {name[7:]: archive.read(name) for name in archive.namelist() if name.startswith("assets/")}
        spec = mujoco.MjSpec.from_string(archive.read("scene.xml").decode(), assets=assets)
        states = times = None
        if "states.npy" in archive.namelist():
            states = np.load(io.BytesIO(archive.read("states.npy")), allow_pickle=False)
            times = np.load(io.BytesIO(archive.read("times.npy")), allow_pickle=False)
            if states.ndim != 2 or times.ndim != 1 or len(states) != len(times) or not len(times):
                raise ValueError("录制数据维度无效")
            if not np.isfinite(states).all() or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                raise ValueError("录制数据含无效数值或时间顺序")
        return spec, manifest, states, times
