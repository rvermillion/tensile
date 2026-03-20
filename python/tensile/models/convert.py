#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import requests
import yaml
import json
from pathlib import Path

from ..common import *
from ..repo import Repo
from .architecture import Architecture
from .load import fetch_hf_config


def convert_config(name: str):
    config_file = Path(name)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        repo_qname = name
        org = None
    else:
        repo = coerce(Repo, f'hf:{name}')
        repo_qname = repo.qname
        config = repo.fetch_config()
        org, name = name.split('/')

    arch = Architecture.from_config(config)

    ten_conf = arch.convert(name, repo_qname, org=org)

    ten_yaml = yaml.safe_dump(ten_conf, sort_keys=False)

    print(ten_yaml)

