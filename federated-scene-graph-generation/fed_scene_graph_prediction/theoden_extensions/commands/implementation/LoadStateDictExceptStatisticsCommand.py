from theoden.operations import LoadStateDictCommand
from theoden.resources import Model


class LoadStateDictExceptStatisticsCommand(LoadStateDictCommand):
    def execute(self):
        # request state dict from server
        model = self._get_files_client_side(self.client_rm)
        # load state dict into model
        sd = self.loader.load(model[self.resource_key])
        self.client_rm.gr(self.resource_key, assert_type=Model).load_state_dict(
            {k: v for k, v in sd.items() if "freq_bias" not in k}, strict=False
        )
