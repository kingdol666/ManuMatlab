import os
from utils import resolve_path

# All paths are relative to the project root

CONFIG_FILE = "config/config.json"
STYLES_FILE = "ui/styles.qss"
ICON_FILE = "ui/public/images/icon.png"
LOGO_FILE = "ui/public/images/logo.svg"
BACKGROUND_IMAGE_FILE = "ui/public/images/background1.png"
CHECK_IMAGE_FILE = "ui/public/images/check.png"
TEMP_MATLAB_DIR = "temp_matlab"
MATLAB_SCRIPTS_DIR = "matlabScripts"

def get_config_file():
    return resolve_path(CONFIG_FILE)

def get_styles_file():
    return resolve_path(STYLES_FILE)

def get_icon_file():
    return resolve_path(ICON_FILE)

def get_logo_file():
    return resolve_path(LOGO_FILE)

def get_background_image_file():
    # Use forward slashes for CSS/QSS paths
    return resolve_path(BACKGROUND_IMAGE_FILE).replace('\\', '/')

def get_check_image_file():
    # Use forward slashes for CSS/QSS paths
    return resolve_path(CHECK_IMAGE_FILE).replace('\\', '/')

def get_temp_matlab_dir():
    return resolve_path(TEMP_MATLAB_DIR)

def get_matlab_scripts_dir():
    return resolve_path(MATLAB_SCRIPTS_DIR)
