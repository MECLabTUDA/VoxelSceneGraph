from theoden.topology.client import Client


class UsernameAwareClient(Client):
    """Client that keeps track of its username."""

    def __init__(
            self,
            communication_address: str = "localhost",
            username: str = "dummy",
            password: str = "dummy",
            ping_interval: float = 1.0
    ) -> None:
        self.username = username
        super().__init__(
            communication_address=communication_address,
            username=username,
            password=password,
            ping_interval=ping_interval,
            rabbitmq=False
        )
