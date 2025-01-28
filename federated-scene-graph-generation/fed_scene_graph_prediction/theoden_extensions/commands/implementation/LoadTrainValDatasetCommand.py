from scene_graph_prediction.data import build_val_data_loaders, build_train_data_loader, save_split
from theoden import Transferable
from theoden.operations import Command


class LoadTrainValDatasetCommand(Command, Transferable):
    def execute(self) -> None:
        # The start_iter is not necessary as the training end is piloted by the server anyway
        logger = self.client_rm["logger"]
        logger.info("Loading train/val datasets")
        train_loader = build_train_data_loader(self.client_rm["cfg"])
        self.client_rm["train_loader_iter"] = iter(train_loader)
        self.client_rm["val_loaders"] = build_val_data_loaders(self.client_rm["cfg"])
        save_split(
            train_loader,
            self.client_rm["val_loaders"],
            None,
            self.client_rm["cfg"].OUTPUT_DIR,
            False
        )
        logger.info("Finished loading train/val datasets")
