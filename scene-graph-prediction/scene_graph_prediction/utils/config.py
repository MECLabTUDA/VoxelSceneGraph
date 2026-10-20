import copy

import yaml
from typing_extensions import Self
from yacs.config import CfgNode, _assert_with_logging, _valid_type


class AccessTrackingCfgNode(CfgNode):
    """
    Yaml configs has the annoying property of having to contain default values for EVERYTHING.
    So even if your method only needs a few parameters, all of them will be saved. Which is very annoying...
    As such, this Cfg?ode class tracks which parameters were accessed i.e. were useful to parametrize your method.
    Now, you can save only what is relevant and produce minimally sized config files.
    """

    def __init__(
            self,
            init_dict: dict | None = None,
            key_list: list[str] | None = None,
            new_allowed: bool = False
    ):
        """
        :param init_dict: the possibly-nested dictionary to initialize the CfgNode.
        :param key_list: a list of names which index this CfgNode from the root.
                         Currently only used for logging purposes.
        :param new_allowed: whether adding new key is allowed when merging with other configs.
        """
        super().__init__(init_dict, key_list, new_allowed)
        self._accessed_keys = set()

    def __getattr__(self, name):
        if name in self:
            self._accessed_keys.add(name)
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, key, value):
        if key == "_accessed_keys":
            return dict.__setattr__(self, key, value)
        return super().__setattr__(key, value)

    def reset_accessed_keys(self):
        self._accessed_keys = set()

    def dump_accessed_only(self, **kwargs):
        """Dump to a string."""
        return yaml.safe_dump(self.convert_to_dict(), **kwargs)

    def convert_to_dict(self) -> dict:
        """Recursively convert the CfgNode to dict."""

        def _convert(cfg_node, key_list):
            if not isinstance(cfg_node, CfgNode):
                _assert_with_logging(
                    _valid_type(cfg_node),
                    f"Key {'.'.join(key_list)} with value {type(cfg_node)} is not a valid type; "
                    "valid types: {_VALID_TYPES}"
                )
                return cfg_node
            else:
                cfg_dict = {k: cfg_node[k] for k in cfg_node._accessed_keys}
                for k, v in cfg_dict.items():
                    cfg_dict[k] = _convert(v, key_list + [k])
                return cfg_dict

        return _convert(self, [])

    def clone_only_accessed(self) -> Self:
        """Returns a copy of the config, but only of keys that were accessed."""
        clone = type(self)(new_allowed=True)
        clone._accessed_keys = copy.deepcopy(self._accessed_keys)
        for key in self._accessed_keys:
            val = self[key]
            if isinstance(val, AccessTrackingCfgNode):
                clone[key] = val.clone_only_accessed()
            else:
                clone[key] = copy.deepcopy(val)
        clone.set_new_allowed(self.is_new_allowed())
        return clone
