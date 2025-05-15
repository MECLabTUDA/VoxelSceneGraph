"""
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

import re as _re

from .ArrayView import RadiologyImageArrayView

VERSION = "0.1.0"


def string_to_regex_pattern(string_filter: str) -> _re.Pattern:
    """
    Converts a raw string to a regex pattern.
    If the regex is invalid, escaped the string.
    """
    if not string_filter:
        # Empty filter
        return _re.compile(".*")
    else:
        # Append ".*" to start and end to match any string containing the string
        # Also check if the pattern is valid
        try:
            return _re.compile(f".*{string_filter}.*")
        except _re.error:
            # Invalid pattern, so escape it
            return _re.compile(f".*{_re.escape(string_filter)}.*")
