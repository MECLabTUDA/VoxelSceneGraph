"""
Here you will find the paths to all assets.
This is here to avoid having to manually track references to files and makes moving / renaming them easier.

Copyright 2023 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys as _sys
from pathlib import Path as _Path

try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    # noinspection PyUnresolvedReferences,PyProtectedMember
    _root_asset_path = _Path(_sys._MEIPASS)
except AttributeError:
    _root_asset_path = _Path("")

_assets = _root_asset_path / "assets"
_assets_common = _assets / "common"
_assets_icons = _assets / "icons"

# Common folder content
logo_path = _assets_common / "logo.png"

# Icons folder content
checkmark_icon = _assets_icons / "checkmark.png"
communication_icon = _assets_icons / "communication.png"
cursor_icon = _assets_icons / "cursor.png"
delete_icon = _assets_icons / "delete.png"
file_icon = _assets_icons / "file.png"
folder_icon = _assets_icons / "folder.png"
luminosity_icon = _assets_icons / "luminosity.png"
magnifier_icon = _assets_icons / "magnifier.png"
merge_icon = _assets_icons / "merge.png"
move_icon = _assets_icons / "move.png"
padding_icon = _assets_icons / "padding.png"
reload_icon = _assets_icons / "reload.png"
scroll_icon = _assets_icons / "scroll.png"
save_icon = _assets_icons / "save.png"
split_icon = _assets_icons / "split.png"
split_cc_icon = _assets_icons / "split_cc.png"
swap_icon = _assets_icons / "swap.png"
zoom_in_icon = _assets_icons / "zoom_in.png"
zoom_out_icon = _assets_icons / "zoom_out.png"
