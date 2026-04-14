
import docker, docker.errors
import os, subprocess
import time
import requests as req


from matplotlib import container

class ContainerManager():
    
    def __init__(self, port: int, container_folder: str, image_name: str) -> None:
        
        self.port = port
        
        self.container_folder = container_folder
        
        self.image_name = image_name
        
        self.docker_file = self.container_folder + "/dockerfile"
        
        self.client = docker.from_env()
        
    
            
    def initialize_container(self):
        
        # First check if an existing image exists then use that, else create a new image 
        
        print("Initializing container on port ", self.port)
        
        print(self.container_folder)
        
        
        ## Check if image exists, if not build it
        ## then check if container is running, if not start it and expose it on the specified port
        self.build_image()
        
        return self.ensure_container_running()
        
    
    def ensure_container_running(self):
    # Check if a container for this image is already running
        containers = self.client.containers.list(
            filters={"ancestor": self.image_name, "status": "running"}
        )
        if containers:
            print(f"Container already running: {containers[0].short_id}")
            return containers[0]
        else:
            time.sleep(2)  # brief pause to ensure image is ready

        print(f"Starting new container from '{self.image_name}' on port {self.port}...")
        container = self.client.containers.run(
            self.image_name,
            detach=True,
            ports={"8000/tcp": self.port},  # adjust internal port as needed
        )
        print(f"Container started: {container.short_id}")
        self.wait_for_ready()   # <-- blocks until /health returns 200
        return container
    
    def wait_for_ready(self, timeout: int = 30, interval: float = 0.5):
        url = f"http://localhost:{self.port}/health"
        deadline = time.time() + timeout
        print(f"Waiting for container to be ready on port {self.port}...")
        while time.time() < deadline:
            try:
                r = req.get(url, timeout=1)
                if r.status_code == 200:
                    print("Container is ready.")
                    return
            except req.exceptions.ConnectionError:
                pass
            time.sleep(interval)
        raise TimeoutError(f"Container on port {self.port} not ready after {timeout}s")        
            
    def build_image(self):
        # Check if image already exists — skip build if it does
        try:
            self.client.images.get(self.image_name)
            print(f"Image '{self.image_name}' already exists, skipping build.")
            return
        except docker.errors.ImageNotFound:
            pass
        
        print(f"Building image '{self.image_name}' from [{self.docker_file}] ...")
        
        try:
            build_logs = self.client.api.build(
                path=self.container_folder,
                tag=self.image_name,
                rm=True,
                decode=True
            )

            for log in build_logs:
                if "stream" in log:
                    print(log["stream"].strip())
                elif "error" in log:
                    print(log["error"].strip())
                    raise Exception(log["error"])

            print(f"Image '{self.image_name}' built successfully.")

        except docker.errors.BuildError as e:
            for line in e.build_log:
                if isinstance(line, dict) and "stream" in line:
                    print(str(line["stream"]).strip())
            print("Error building image: ", e)
            raise
        
        except docker.errors.APIError as e:
            print(f"Docker API error during build: {e}")
            raise
            
        
    def check_container(self):
        pass