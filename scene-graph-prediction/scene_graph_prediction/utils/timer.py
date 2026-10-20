# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import datetime
import time


class Timer:
    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.reset()

    @property
    def average_time(self) -> float:
        return self.total_time / self.calls if self.calls > 0 else 0.0

    def tic(self):
        """Sets the start time."""
        # using time.time instead of time.clock because time.clock does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average: bool = True) -> float:
        """
        Computes the time difference to the last call to tic().
        If average is True, returns the average difference across each calls to the timer.
        Else returns the current time difference.
        """
        self.add(time.time() - self.start_time)
        if average:
            return self.average_time
        else:
            return self.diff

    def add(self, time_diff: float):
        self.diff = time_diff
        self.total_time += self.diff
        self.calls += 1

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0

    def avg_time_str(self) -> str:
        return self.timedelta_str(self.average_time)

    @staticmethod
    def timedelta_str(seconds_diff: float) -> str:
        return str(datetime.timedelta(seconds=seconds_diff))
