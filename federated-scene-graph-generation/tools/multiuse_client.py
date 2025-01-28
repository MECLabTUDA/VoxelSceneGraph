from getpass import getpass
from pathlib import Path

import click

from fed_scene_graph_prediction.theoden_extensions import UsernameAwareClient
from theoden import GlobalContext


@click.command()
@click.option("-c", "--global-context",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
              required=False, default=None, help="Optional path to the global context.")
@click.option("-a", "--communication-address",
              type=str, required=True, help="IP address of the server.")
@click.option("-u", "--username",
              type=str, required=False, default="dummy", help="Client username.")
@click.option("-p", "--password",
              type=str, required=False, default="dummy", help="Client password.")
def main(global_context: Path | None, communication_address: str, username: str, password: str) -> int:
    # Load the global context if it is provided
    if global_context is not None:
        GlobalContext().load_from_yaml(global_context.as_posix())

    # If the username is not "dummy" and the password is "dummy", prompt the user for the password
    if username != "dummy" and password == "dummy":
        password = getpass("Password: ")

    # Start the client
    client = UsernameAwareClient(
        communication_address=communication_address,
        username=username,
        password=password,
        ping_interval=1.
    )
    client.start()

    return 0


if __name__ == "__main__":
    exit(main())
