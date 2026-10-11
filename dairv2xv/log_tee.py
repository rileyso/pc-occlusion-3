"""Mirror stdout to a log file, flushed per line, so a watcher can tail it."""

import sys


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def tee_to_log(path):
    """Also mirror stderr, so a crash traceback reaches the log."""
    log = open(path, "w")
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)
