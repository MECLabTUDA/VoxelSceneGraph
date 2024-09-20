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

import json as _json
import os as _os
from logging import Logger as _Logger
from typing import Any as _Any, Iterable

from jsonschema.validators import Draft202012Validator
from referencing import Resource, Registry
from referencing.jsonschema import DRAFT202012

ID_SCHEMA_NAME = "Id"


def schema_to_resource(schema: dict) -> Resource:
    """Convert a schema into a resource to add to a Registry."""
    return DRAFT202012.create_resource(schema)


def get_validator(schema: dict, registry: Registry | None) -> Draft202012Validator:
    """Get the proper validator to validate JSON data."""
    # noinspection PyArgumentList
    return Draft202012Validator(schema, registry=registry)


def id_schema() -> dict:
    """Return the schema for an id."""
    return {"id": ID_SCHEMA_NAME, "type": "integer", "minimum": 1}


def list_of_names_to_pattern_filter(names: Iterable[str]) -> str:
    """Converts a list of names to a regex that only accepts one those."""
    return "^" + "|".join(f"(?:{name})" for name in names) + "$"


def expect_str(value: _Any, context_str: str, logger: _Logger) -> str:
    """
    Checks that the value is a string or attempts to cast it
    :param value: a value of any type that is expected to be a string or that should be cast
    :param context_str: context given by context(...) or any string
    :param logger: the logger
    :return: None if the value is not a string or the cast failed, else a string
    """
    if isinstance(value, str):
        return value
    logger.warning(f"{context_str} value \"{value}\" is expected to be a string. Casting value to str...")
    return str(value)


def expect_int(value: _Any, context_str: str, logger: _Logger) -> int | None:
    """
    Checks that the value is an int or attempts to cast it
    :param value: a value of any type that is expected to be an int or that should be cast
    :param context_str: context given by context(...) or any string
    :param logger: the logger
    :return: None if the value is not an int or the cast failed, else an int
    """
    if isinstance(value, int):
        return value
    logger.warning(f"{context_str} value \"{value}\" is expected to be an int. Casting value to int...")
    try:
        return int(value)
    except ValueError:
        logger.error(f"{context_str} cast to int failed.")


def expect_float(value: _Any, context_str: str, logger: _Logger) -> float | None:
    """
    Checks that the value is a float or attempts to cast it
    :param value: a value of any type that is expected to be a float or that should be cast
    :param context_str: context given by context(...) or any string
    :param logger: the logger
    :return: None if the value is not a float or the cast failed, else a float
    """
    if isinstance(value, float):
        return value
    try:
        # Some floats can also be represented as a string
        # So we attempt to cast to float and if it fails, we warn and error
        return float(value)
    except ValueError:
        logger.warning(f"{context_str} value \"{value}\" is expected to be an float. Casting value to float...")
        logger.error(f"{context_str} cast to float failed.")


def expect_bool(value: _Any, context_str: str, logger: _Logger) -> bool:
    """
    Checks that the value is a bool or attempts to cast it
    :param value: a value of any type that is expected to be a bool or that should be cast
    :param context_str: context given by context(...) or any string
    :param logger: the logger
    :return: None if the value is not a bool or the cast failed, else a bool
    """
    if isinstance(value, bool):
        return value
    logger.warning(f"{context_str} value \"{value}\" is expected to be a bool. Casting value to bool...")
    return bool(value)


def expect_list(value: _Any, context_str: str, logger: _Logger) -> list | None:
    """
    Checks that the value is a list
    :param value: a value of any type that is expected to be a list
    :param context_str: context given by context(...) or any string
    :param logger: the logger
    :return: None if the value is not a list, else a list
    """
    if isinstance(value, list):
        return value
    logger.error(f"{context_str} value \"{value}\" is expected to be a list.")
    return None


def expect_dict(value: _Any, context_str: str, logger: _Logger) -> dict | None:
    """
    Checks that the value is a dict
    :param value: a value of any type that is expected to be a dict
    :param context_str: context given by context(...) or any string
    :param logger: the logger
    :return: None if the value is not a dict, else a dict
    """
    if isinstance(value, dict):
        return value
    logger.error(f"{context_str} value \"{value}\" is expected to be a dict.")
    return None


def check_list_unicity(
        value_list: list,
        logger: _Logger,
        value_name: str = "Value",
        warn_only: bool = False
) -> bool:
    """Function used to check that a list of id contains each id only one time"""
    # List should be small enough that using a list is faster than using a set
    values_seen = []
    success = True
    for val in value_list:
        if val not in values_seen:
            values_seen.append(val)
        else:
            (logger.warning if warn_only else logger.error)(f"{value_name} \"{val}\" is not unique.")
            success = False
    return success or warn_only


def load_json_from_path(path: str | _os.PathLike,
                        context_str: str | None = None,
                        logger: _Logger | None = None) -> dict | None:
    """Attempts to read a json file from a path and then attempts to load the specified object_class from the dict."""
    if not _os.path.isfile(path):
        if logger and context_str is not None:
            logger.error(f"{context_str} File not found.")
        return

    with open(path, "r") as f:
        try:
            return _json.load(f)
        except (_json.decoder.JSONDecodeError, UnicodeDecodeError):
            # UnicodeDecodeError for files that don't contain text
            # JSONDecodeError for invalid JSON files
            f.close()
            # _os.remove(path)
            logger.error(f"{context_str} Invalid file content.")
