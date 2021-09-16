import os

import cv2

import helpers
import rotate

IMAGE_FOLDER = os.path.abspath(r'C:\Users\admin\Desktop\detect\dataset')
ROTATED_FOLDER = os.path.abspath(r'C:\Users\admin\Desktop\detect\rotated_dataset')
ANGLE_VALUE = 10

for root, dirs, files in os.walk(IMAGE_FOLDER):
    for f in files:
        if f[-3:] == "jpg":
            origin_file = os.path.join(root, f)
            ## C:\Users\admin\Desktop\detect\dataset\1.jpg
            origin_file_name = origin_file.split("\\")[-1].split(".")[0]
            print(origin_file_name)

            for angle in range(0, 360, ANGLE_VALUE):
                yolo_set = rotate.yoloRotatebbox(origin_file.split(".")[0], '.jpg', angle)
                ## origin_file.split(".")[0]  -> C:\Users\admin\Desktop\detect\dataset\1
                rotated_box = yolo_set.rotateYolobbox()
                rotated_image = yolo_set.rotate_image()

                new_file_name = origin_file_name + "_" + str(angle)
                new_image_name = new_file_name + ".jpg"
                new_box_name = new_file_name + ".txt"
                
                new_image_path = os.path.join(ROTATED_FOLDER, new_image_name)
                new_box_path = os.path.join(ROTATED_FOLDER, new_box_name)

                cv2.imwrite(new_image_path, rotated_image)

                if os.path.exists(new_box_path):
                    os.remove(new_box_path)

                for i in rotated_box:
                    with open(new_box_path, 'a') as box:
                        box.writelines(' '.join(map(str, helpers.cvFormattoYolo(i, yolo_set.rotate_image().shape[0], yolo_set.rotate_image().shape[1]))) + '\n')
