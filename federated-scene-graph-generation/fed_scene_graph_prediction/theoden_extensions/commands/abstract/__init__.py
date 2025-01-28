from theoden.common import Transferable
from theoden.operations.commands import AbstractCommand


class ABCInitClientCommand(AbstractCommand, Transferable):
    """
    Initialize the client by:
    - Loading the config sent by the server
    - Adapting it by replacing keys using the federated client config for its client name
    - Creating the model structure
    """

    def __init__(self, cfg: dict, uuid: str | None = None):
        super().__init__(uuid=uuid)
        self.cfg = cfg  # Main server config node as dict
