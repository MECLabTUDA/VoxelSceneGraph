from theoden import ExecutionResponse
from theoden.operations import Command
from theoden.resources import ResourceManager
from theoden.topology import Topology


class SendArgumentsCommand(Command):
    """Send the training arguments dict from scene_graph_prediction to the server."""

    def execute(self) -> ExecutionResponse | None:
        logger = self.client_rm["logger"]
        logger.info("Sending arguments dict to the server")
        return ExecutionResponse(data={"arguments": self.client_rm["arguments"]})

    def on_init_server_side(
            self,
            topology: Topology,
            resource_manager: ResourceManager,
            selected_clients: list[str],
    ) -> None:
        resource_manager.sr("arguments", {})

    def on_client_finish_server_side(
            self,
            topology: Topology,
            resource_manager: ResourceManager,
            client_name: str,
            execution_response: ExecutionResponse,
            instruction_uuid: str,
    ):
        # We don't care that each client will overwrite each other
        resource_manager.sr("arguments", execution_response.data["arguments"])
