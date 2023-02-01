import argparse
import glob
from tqdm import tqdm
import os
import cv2
import logging
import numpy as np

from config import Config

import my_utils as mu
from mediapipe_face_detection import detect_face


logging.basicConfig(level = logging.INFO)


def read_save_video(
    params:dict
):
    ## reading video file
    config = params["config"]

    base_path = config["dataset"]["path"]
    new_path = config["dataset"]["new_path"]

    video_directory = params["video_directory"]
    save_every = params["save_every"]

    
    cap = cv2.VideoCapture(os.path.join(base_path, video_directory, config["dataset"]["video_file_name"]))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=length, ascii=True)
    # current frame index in image
    counter=0
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret:
                counter+=1
                pass
                
                if (counter%save_every)==0:

                    record_file_path = config["dataset"]["log_file"]

                    # face detection and crop image
                    # we should add padding to face detection bbox values to include more landmark 
                    face_bbox = detect_face(frame)
                    
                    if not face_bbox:
                        message = f"Could not detect face on frame {video_directory}/{counter}(image_{counter}.png)"
                        tqdm.write(message)
                        mu.save_file_info_txt(record_file_path, message)
                        continue


                    # apply padding
                    face_bbox = mu.apply_bbox_padding(face_bbox, config["dataset"]["face_detection_padding"], frame.shape[:2])

                    x_min, y_min, x_max, y_max = face_bbox
                    cropped_image = frame[y_min:y_max, x_min:x_max]


                    points = mu.read_image_points(base_path, video_directory, counter)
                    relative_points = mu.relative_cropped_points(points, face_bbox)


                    image_name = f"image_{counter}.png"
                    # save cropped image to directory
                    res_img=save_image(
                        cropped_image,
                        {
                            "base_path": new_path,
                            "inner_directory": video_directory,
                            "image_name": image_name
                        }
                    )
                    points_name = f"points_{counter}.npy"

                    res_points=mu.save_numpy(
                        relative_points,
                        os.path.join(new_path, video_directory, points_name) 
                    )
                    file_write_text = None    

                    
                    previous_path = os.path.join(base_path,video_directory,config['dataset']['video_file_name'])
                    saved_path = os.path.join(new_path,video_directory,image_name)

                    if not (res_img and res_points):
                        tqdm.write(f"Either image or points are not successfuly written!! or both! --> '{video_directory}'-'{image_name}'-'{points_name}' to '{os.path.join(new_path, video_directory)}'")

                        # assert (res_img and res_points), "Either image or points are not successfuly written!! or both!"

                        file_write_text = f"Fail on save {previous_path} to {saved_path}"

                    else: 
                        # success
                        file_write_text = f"Saved {previous_path} to {saved_path}"


                    mu.save_file_info_txt(record_file_path, file_write_text)
                    #logging.info(f"Saved image {image_name} to {os.path.join(new_path, video_directory)}")
                    tqdm.write(f"File info written for image {video_directory} {image_name} to {record_file_path}")
                    
                
                    
                
            else:
                #logging.info(f"Finished processing video file {image_name}")
                tqdm.write(f"Finished processing video file {image_name}")
                break         
        except:
            error_message = f"Error processing video file!!"
            logging.exception(error_message)
        finally:
            pbar.update(1)
    print(counter, length)
    pbar.close()
            

def save_image(
    image:np.ndarray,
    path_params:dict
):

    base_path = path_params["base_path"]
    inner_directory = path_params["inner_directory"]
    image_name = path_params["image_name"]

    directory = os.path.join(base_path, inner_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


    save_path = os.path.join(base_path, inner_directory, image_name)
    try:
        cv2.imwrite(save_path, image)
        return True
    except:
        logging.exception(f"Error saving image {image_name}")







def create_dataset(
    config:dict,
    save_every:int
):
    # Deconstruct path config
    base_path = config["dataset"]["path"]

    logging.info("Processing Dataset...")
    for video_directory_path in tqdm(glob.glob(os.path.join(base_path, "*")), leave=False, ascii=True):


        # video_directory_path --> <base_path>/<video_directory>
        video_directory = video_directory_path.split(os.sep)[-1]
        # video_directory --> 001, 002, ..., 208, ... 

        if video_directory in config["dataset"]["ignore_files"]:
            logging.info(f"Ignored {video_directory}!!")
            continue
        
        
        params={
            "config": config,
            "save_every": save_every,
            "video_directory": video_directory
        }
        
        read_save_video(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yaml")

    args = parser.parse_args()
    
    cfg = Config(args.config)

    print(cfg.config)

    create_dataset(cfg.config, cfg.config["dataset"]["save_every"])

