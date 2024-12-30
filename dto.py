from typing import Protocol
from dataclasses import dataclass, fields
from abc import ABC


class ConfigurationClass:
    def __init__(self, cfg: dict):
        for field, value in zip(fields(self), cfg.values()):
            setattr(self, field.name, value)


@dataclass
class DatasetCreationConfiguration(ConfigurationClass):
    """ Train hyperparameters dto"""
    dataset_path: str
    dataset_out_path: str
    ignore_files: list
    video_file_name: str
    save_every: int
    face_detection_padding: 20
    log_file: str
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)