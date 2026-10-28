"""Optional read-only training-start gallery; never consumes policy RNG state."""
import hashlib
import json
from pathlib import Path
import secrets


class ReferenceCatalog:
    def __init__(self, manifest, task, instruction):
        self.path = Path(manifest).expanduser().resolve()
        payload = json.loads(self.path.read_text())
        if payload.get("task") != task or payload.get("instruction") != instruction:
            raise ValueError("Reference task/instruction differs from this deployment")
        self.samples = payload["samples"]
        if not self.samples or len(self.samples) != payload["episode_count"]:
            raise ValueError("Reference episode collection is incomplete")
        self.index = 0
        for sample in self.samples:
            if str(sample.get("step")) not in {"0", "0000"}:
                raise ValueError("Reference must be an episode start frame")
            for role in ("primary", "wrist"):
                path = self.image_path(sample, role)
                expected = sample[role].get("sha256")
                if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Reference checksum mismatch: {path}")

    def image_path(self, sample, role):
        path = (self.path.parent / sample[role]["filename"]).resolve()
        if not path.is_relative_to(self.path.parent):
            raise ValueError("Reference path escaped its task directory")
        return path

    def move(self, offset):
        self.index = (self.index+offset) % len(self.samples)

    def randomize(self):
        self.index = secrets.randbelow(len(self.samples))

    def images(self):
        from PIL import Image
        output = []
        for role in ("primary", "wrist"):
            with Image.open(self.image_path(self.samples[self.index], role)) as image:
                output.append(image.convert("RGB"))
        return output
