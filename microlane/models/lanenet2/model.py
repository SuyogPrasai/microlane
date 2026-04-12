# Basically what this is going to do is
# Check if we have a docker container that is already running
# If we do then we use that
# If we don't then we instantiate a new docker container

# Then, we will have different functiosn for contacting the different api's of the container
# predict one, batch predict, etc

from microlane.models.lanenet2.container import ContainerManager

class LaneNet2():
    
    def __init__(self, container_folder: str, image_name: str, port = 8000) -> None:
        
        self.container_port = port
        
        self.container_folder = container_folder
        
        self.image_name = image_name

        self.container_manager = ContainerManager(
            
            port=self.container_port, 
            
            container_folder=self.container_folder,
            
            image_name=self.image_name
        )
        
        self.container_manager.initialize_container()