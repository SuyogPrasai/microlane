import time
import requests as req 

from microlane.utils.load_config import load_config
from microlane.utils.container import ContainerManager
from microlane.schemas.config import ModelConfig

class Model:
    
    def __init__(self) -> None:
        
        self.config = load_config()
        
        self.container_manager = ContainerManager()
        
        self.active_containers = self.container_manager.active_containers()
        
        self.active_images = self.container_manager.active_images()
            
    
    def stop(self, container_id) -> None:
        self.container_manager.stop_container(container_id=container_id)
        
    def restart(self, container_id) -> None:
        self.container_manager.restart_container(container_id=container_id)
    
    def _initialize_container(self, model: ModelConfig) -> str:

        for container in self.active_containers:
            if (
                container.image
                and container.id
                and container.image.tags
                and model.image_name in container.image.tags
            ):
                if container.status == "running":
                    print(f"Container {container.id} is already running.")
                    return container.id
                else:
                    try:
                        print(f"Container {container.id} is not running. Restarting...")
                        self.container_manager.restart_container(container_id=container.id)
                        return container.id
                    except Exception as e:
                        print(f"Failed to restart container {container.id}: {e}")
                        continue

        image_exists = any(
            image.tags and model.image_name in image.tags
            for image in self.active_images
        )

        if image_exists:
            print(f"Image {model.image_name} found. Starting container...")
            try:
                container = self.container_manager.start_container(
                    image_name=model.image_name,
                    port=model.port
                )
                if container.id is None:
                    raise RuntimeError("Container started but no ID was returned.")
                return container.id
            except Exception as e:
                raise RuntimeError(f"Failed to start container: {e}")

        else:
            print(f"Image {model.image_name} not found. Building image...")
            try:
                self.container_manager.build_image(
                    dockerfile_path=model.container_folder / "Dockerfile",
                    image_name=model.image_name
                )
                container = self.container_manager.start_container(
                    image_name=model.image_name,
                    port=model.port
                )
                if container.id is None:
                    raise RuntimeError("Container started but no ID was returned.")
                return container.id
            except Exception as e:
                raise RuntimeError(f"Failed to build image or start container: {e}")       

    def initialize_model(self, model: ModelConfig) -> str:
        try:
            container_id = self._initialize_container(model)
            
            self.wait_for_ready(
                model.port
            )
            
            return container_id
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
    
    
    def wait_for_ready(self, port, timeout: int = 30, interval: float = 0.5):

        url = f"http://localhost:{port}/health"

        deadline = time.time() + timeout

        print(f"Waiting for container to be ready on port {port}...")

        while time.time() < deadline:

            try:
                r = req.get(url, timeout=1)

                if r.status_code == 200:

                    print("Container is ready.")
                    
                    return

            except req.exceptions.ConnectionError:

                pass

            time.sleep(interval)

        raise TimeoutError(f"Container on port {port} not ready after {timeout}s")        