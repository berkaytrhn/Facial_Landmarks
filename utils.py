
import numpy as np
from config import Config
import cv2
import json



# Used in older versions, not currently
def join_path(*directories):
    return "/".join(directories)


def save_file_info_txt(
    save_path:str,
    content:str
):
    with open(save_path, "a") as _file:
        _file.write(f"{content}\n")

def apply_bbox_padding(
    bbox_coordinates:tuple,
    padding:int,
    image_shape:tuple
):
    height, width = image_shape

    x_min, y_min, x_max, y_max = bbox_coordinates

    x_min=(x_min-padding) if ((x_min-padding) > 0) else 0
    y_min=(y_min-padding) if ((y_min-padding) > 0) else 0
    x_max=(x_max+padding) if ((x_max+padding) < width) else (width-1)
    y_max=(y_max+padding) if ((y_max+padding) < height) else (height-1)

    return (x_min, y_min, x_max, y_max)


def load_numpy(path:str):
    try:
        array = np.load(path)
        return array
    except:
        return None

def save_numpy(
    array:np.ndarray, 
    path:str
):
    try:
        np.save(path, array)
        return True
    except:
        return False

def read_json(
    json_path:str):
    with open(json_path, "r") as json_file:
        json.load(json_file)


def write_json(
    dictionary:dict, 
    path:str
):
    with open(path, "w") as json_file:
        json.dump(dictionary, json_file)

def relative_cropped_points(
    points:list, 
    crop_coordinates:tuple
):
    relative_coords=list()
    x_min, y_min, x_max, y_max = crop_coordinates
    for coords in points:
        coord_x, coord_y = coords
        relative_x = int(coord_x-x_min)
        relative_y = int(coord_y-y_min)
        relative_coords.append((relative_x, relative_y))
    return np.array(relative_coords)


def draw_landmarks(
    image:np.ndarray, 
    points:np.ndarray
):

    temp_image = image.copy()
    points=points.astype(np.int32)
    for point_pair in points:
        # point_pair ---> np.ndarray  [<number_1>, <number_2>]
        temp_image = cv2.circle(temp_image, tuple(point_pair), 1, (255,0,0), 2)
    return temp_image

def get_category_from_directory(
    directory:str,
    category_config:dict
):

    for category, values in category_config.items():
        if int(directory) in values:
            return category


def read_image_points(
    base_path:str, # ex: '300VW_Dataset_2015_12_14/300VW_Dataset_2015_12_14'
    video_number:int, # ex: 1
    index:int, # ex: 13
):
    points = np.loadtxt(
        join_path(
            base_path, 
            str(video_number).zfill(3), 
            "annot", 
            f"{str(index).zfill(6)}.pts"
        ),
        comments=(
            "version:", 
            "n_points:", 
            "{", "}"
        )
    )
    return points


