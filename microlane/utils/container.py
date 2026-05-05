import docker
from docker.models.containers import Container
from docker.models.images import Image

from pathlib import Path
from typing import List, Tuple
from microlane.utils.load_config import load_config

config = load_config()

class ContainerManager:

    def __init__(self) -> None:
        self.client = docker.from_env()

    def start_container(self, image_name: str, port: int):
        """Create and start a new container from the given image."""
        container = self.client.containers.run(
            image_name,
            detach=True,
            ports={f"{config.pipeline.default_port}/tcp": port},
        )
        return container

    def stop_container(self, container_id: str):
        """Stop a running container by ID."""
        container = self.client.containers.get(container_id)
        container.stop()

    def restart_container(self, container_id: str):
        """Restart a container by ID."""
        container = self.client.containers.get(container_id)
        container.restart()

    def build_image(self, dockerfile_path: Path, image_name: str):
        """Build a Docker image from a Dockerfile."""
        all_logs = []
        for log in self.client.api.build(
            path=str(dockerfile_path.parent),
            dockerfile=str(dockerfile_path.name),
            tag=image_name,
            rm=True,
            decode=True
        ):
            all_logs.append(log)
            if "stream" in log:
                print(log["stream"], end="", flush=True)
            elif "error" in log:
                print(log["error"], flush=True)
                raise Exception(log["error"])

        image = self.client.images.get(image_name)
        return image, all_logs

    def active_images(self) -> List[Image]:
        """Return a list of all locally available images."""
        return self.client.images.list(all=True)

    def active_containers(self) -> List[Container]:
        """Return a list of all currently running containers."""
        return self.client.containers.list(all=True)