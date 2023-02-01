import yaml
import traceback

class Config:
    def __init__(
        self, 
        config_file_path:str
    ):
        self.config = self.read_config(config_file_path)
        assert self.config != None, "Config file not loaded!"

    def read_config(
        self,
        config_file:str
    ):
        try:
            with open(config_file, "r") as _file:
                config = yaml.safe_load(_file)
                return config
        except:
            error_string = f"Error loading config file in {self.__class__}\n\n{traceback.format_exc()}"
            print(error_string)
            return None


if __name__ == "__main__":
    Config("config.yaml")