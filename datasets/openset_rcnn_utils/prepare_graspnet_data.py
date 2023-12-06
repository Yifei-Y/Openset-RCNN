import os, argparse, glob, time
from shutil import copyfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to dataset directory')
    parser.add_argument('--image_destination', type=str, help='Path to store images.')
    args = parser.parse_args()

    folders = ['train', 'test_seen', 'test_similar', 'test_novel']
    for folder in folders:
        scene_path = os.path.join(args.dataset_path, folder)
        scene_list = sorted(glob.glob(scene_path + '/scene_*'))
        for scene in scene_list:
            print(scene)
            image_files = sorted(glob.glob(os.path.abspath(scene + '/realsense/rgb/[0-9][0-9][0-9][0-9].png')))
            for image in image_files:
                current_number = len(os.listdir(args.image_destination))
                print('current image number : {}'.format(current_number))
                destination_filename = args.image_destination + '/{:06d}'.format(current_number + 1) + '.png'
                copyfile(image, destination_filename)
                time.sleep(0.05)
    print('finish!')