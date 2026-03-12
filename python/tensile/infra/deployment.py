#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from os import environ

debug: bool = True

debug = environ.get('DEBUG', str(debug)).lower() == 'true'
