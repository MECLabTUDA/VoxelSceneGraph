from fed_scene_graph_prediction.theoden_extensions.actions import SaveCheckpointAction
from theoden.common import Transferable
from theoden.operations import Aggregator, ClosedDistribution, BinarySelector, InstructionBundle, \
    DefaultAggregationBundle
from theoden.operations.commands import LoadStateDictCommand, SequentialCommand, TrainValNTimesCommand, \
    CalculateClientScoreCommand, ValidateEpochCommand
from theoden.resources import StateLoader
from .implementation import SendArgumentsCommand


class SceneGraphPredictionInstructionBundle(InstructionBundle, Transferable):
    """Adaptation of the MultiRoundTrainingInstructionBundle for the scene_graph_prediction pipeline."""

    def __init__(
            self,
            n_rounds: int,
            aggregator: Aggregator,
            epochs_per_round: int | None = None,
            steps_per_round: int | None = None,
            selector: BinarySelector | None = None,
            train_batch_size: int = 32,
            validation_batch_size: int = 32,
            num_workers: int = 6,
            client_score_command: CalculateClientScoreCommand | None = None,
            final_validation: bool = True,
            val_split: str = "val",
            start_at_round: int = 0,
            validate_every_n_rounds: int | None = None,
            start_n_rounds_without_validation: int = 0,
            label_key: str = "class_label",
            loader: type[StateLoader] | None = None,
            only_grad: bool = False,
            simultaneous_execution: int = 0,
    ) -> None:
        """The MultiRoundTrainingInstructionBundle is a convenience class for creating a sequence of instructions that train a model for a given number of rounds.

        Args:
            n_rounds (int): The number of rounds to train the model for.
            aggregator (Aggregator): The aggregator to use for aggregating the model updates.
            epochs_per_round (int, optional): The number of epochs to train the model for in each round. Defaults to None.
            steps_per_round (int, optional): The number of steps to train the model for in each round. Defaults to None.
            final_validation (bool, optional): Whether to perform a final validation on the test set after training. Defaults to True.
            val_split (str, optional): The split to use for validation. Defaults to "val".
            start_at_round (int, optional): The round to start at. Defaults to 0.
            validate_every_n_rounds (int, optional): Whether to validate after every n rounds. Defaults to None.
            start_n_rounds_without_validation (int, optional): The number of rounds to start without validation. Defaults to 0.
            label_key (str, optional): The key of the label in the dataset. Defaults to "class_label".
            simultaneous_execution (int, optional): The number of client that simultaneous executions of a command. Defaults to 0 (all clients).

        Raises:
            ValueError: If epochs_per_round and steps_per_round are both None or both not None.
        """
        instructions = []
        for i in range(n_rounds):
            validate = (
                           i % validate_every_n_rounds == 0 and i > 0
                           if validate_every_n_rounds
                           else True
                       ) and i >= start_n_rounds_without_validation

            if validate:
                # 6. Send scene_graph_prediction training arguments to the server post-validation
                instructions.append(ClosedDistribution(commands=[SendArgumentsCommand()]))
                # 7. Let the server save the checkpoint
                instructions.append(SaveCheckpointAction(is_final_save=False))

            # 1. Send model, optimizer, and scheduler to clients
            # 2. Train
            # 3. Send model, optimizer, and scheduler to the server
            # 4. Score clients if client_score_command
            # 5. Aggregate
            instructions.append(
                DefaultAggregationBundle(
                    selector=selector,
                    train_command=TrainValNTimesCommand(
                        n_epochs=epochs_per_round,
                        n_steps=steps_per_round,
                        val_split=val_split,
                        communication_round=start_at_round + i + 1,
                        # We need to do the validation directly after the aggregation, but not on the first round
                        start_with_val=True,
                        end_with_val=False,
                        label_key=label_key,
                        train_batch_size=train_batch_size,
                        validation_batch_size=validation_batch_size,
                        num_workers=num_workers,
                        validate=validate
                    ),
                    client_score_command=client_score_command,
                    aggregator=aggregator,
                    simultaneous_execution=simultaneous_execution,
                    loader=loader,
                    only_grad=only_grad,
                    model_keys=["model"],
                )
            )

        if final_validation:
            # -2. Perform a validation round on the test split
            instructions.append(
                ClosedDistribution(
                    SequentialCommand(
                        [
                            LoadStateDictCommand(
                                "model",
                                checkpoint_key="__global__",
                                loader=loader,
                            ),
                            SendArgumentsCommand(),
                            ValidateEpochCommand(
                                split="val",
                                label_key=label_key,
                                batch_size=validation_batch_size,
                                num_workers=num_workers,
                            ),
                            ValidateEpochCommand(
                                split="test",
                                label_key=label_key,
                                batch_size=validation_batch_size,
                                num_workers=num_workers,
                            ),
                        ]
                    ),
                    simultaneous_execution=simultaneous_execution,
                )
            )
            # 1. Let the server save the final checkpoint
            instructions.append(SaveCheckpointAction(is_final_save=True))
        super().__init__(instructions)
