from mne import Report as MneReport
from exca import ConfDict
from yaml import safe_load
from . import paths

import inspect, datetime
from pathlib import Path

class Report(MneReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, fname=None, *args, open_browser=False, overwrite=True, **kwargs):
        if fname is None:
            fname = inspect.stack()[1]
            importer_file = Path(fname.filename).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{importer_file}_{timestamp}.html"
        super().save(paths.results / fname, *args, open_browser=open_browser, overwrite=overwrite, **kwargs)

def read_config(path):
    with open(path, "r") as f:
        content = f.read()
    return ConfDict(safe_load(content))