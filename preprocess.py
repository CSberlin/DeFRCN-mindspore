import os
import time
import numpy as np
import sys
from src.dataset import create_defrcn_dataset, data_to_mindrecord_byte_image
from src.model_utils.config import config


def main():
    prefix = "mindrecord"
    mindrecord_dir = os.path.join(config.test_mindrecord_dir, config.dataset_test[0])
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(mindrecord_file + ".db"):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, config.dataset_test, prefix, False, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        print("wait 5 seconds")
        time.sleep(5)

    dataset = create_defrcn_dataset(config, mindrecord_file, batch_size=config.test_batch_size, is_training=False, num_parallel_workers=1)
    it = dataset.create_dict_iterator(output_numpy=True)

    data_path = config.input_path
    os.makedirs(data_path, exist_ok=True)

    img_dir = os.path.join(data_path, "eval_img_bin")
    os.makedirs(img_dir)

    img_id_dir = os.path.join(data_path, "eval_id_bin")
    os.makedirs(img_id_dir)

    img_meta_dir = os.path.join(data_path, "eval_meta_bin")
    os.makedirs(img_meta_dir)

    # Record the shape, because when numpy read binary file(np.fromfile()), the shape should be given.
    # Otherwise, the data would be wrong
    shape_recorder = open(os.path.join(data_path, "eval_shapes"), 'w')

    for i,data in enumerate(it):
        
        input_img_file = "eval_input_" + str(i + 1) + ".bin"
        input_img_id_file = "eval_input_" + str(i + 1) + ".bin"
        input_img_meta_file = "eval_input_" + str(i + 1) + ".bin"
        
        img = data['image']
        img_id = data['image_id']
        img_meta = data['image_shape']
        
        input_img_path = os.path.join(img_dir, input_img_file)
        input_img_id_path = os.path.join(img_id_dir, input_img_id_file)
        input_img_meta_path = os.path.join(img_meta_dir, input_img_meta_file)
        
        img.tofile(input_img_path)
        img_id.tofile(input_img_id_path)
        img_meta.tofile(input_img_meta_path)

        shape_recorder.write(str(data['image_shape'].shape) + "\n")
        shape_recorder.write(str(data['image_id'].shape) + "\n")
        shape_recorder.write(str(data['image'].shape) + "\n")

    shape_recorder.close()
    print("output preprocess data finished")

if __name__ == '__main__':
    main()