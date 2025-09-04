import yaml
import os

class ConfigManager:
    def __init__(self, 
                 config_path: str | None = None):
        if config_path is None:
            root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(root_path, 'config.yaml')
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_all(self):
        return self.config

