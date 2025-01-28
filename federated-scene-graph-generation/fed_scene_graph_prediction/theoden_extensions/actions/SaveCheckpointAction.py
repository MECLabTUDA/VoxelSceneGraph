from theoden.operations import Action
from theoden.resources import ResourceManager
from theoden.resources.meta import DictCheckpoint
from theoden.topology import Topology


class SaveCheckpointAction(Action):
    """Save the current checkpoint using the scene_graph_prediction framework."""

    def __init__(self, is_final_save: bool = False):
        super().__init__()
        self.is_final_save = is_final_save

    def perform(self, topology: Topology, resource_manager: ResourceManager):
        """Set the state dict of all clients to a unified state dict.

        Return a successor instruction, that will be executed after the model is initialized.
        This Instruction will select a state dict and distribute it to all clients.

        Args:
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.

        Returns:
            Instruction: The successor instruction.
        """
        # Get model state
        # noinspection PyTypeChecker
        model: DictCheckpoint = resource_manager.checkpoint_manager.get_checkpoint(
            resource_type="model", resource_key="model", checkpoint_key="__global__"
        )

        # Get the checkpointer
        checkpointer = resource_manager["checkpointer"]
        checkpointer.model.load_state_dict(model.data)

        # Get arguments from any of the clients
        arguments = resource_manager.gr("arguments")

        # Save the model
        if self.is_final_save:
            checkpointer.save("model_final", **arguments)
        else:
            checkpointer.save(f"model_{arguments['iteration']:07d}", **arguments)
