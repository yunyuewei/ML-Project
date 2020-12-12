import os
import random
import shutil

if __name__ == "__main__":
    dir_data = 'train_data_v2/'
    files = os.listdir(dir_data)
    images = {image.split('.')[0]: [] for image in files}
    images = list(images.keys())

    matched_data = {str(x): [] for x in range(40)}

    for i, img in enumerate(images):
        assert os.path.isfile(
            dir_data + '/' + img+'.txt'), "{} does not exist!".format(dir_data + '/' + img+'.txt')
        assert os.path.isfile(
            dir_data + '/' + img+'.jpg'), "{} does not exist!".format(dir_data + '/' + img+'.jpg')

        with open(dir_data + '/' + img+'.txt', 'r') as fi:
            label = fi.readline().strip()
            label = label.split(' ')[-1]
            print("\r[{}/{}]".format(i+1, len(images)), end='')
            matched_data[label].append(dir_data + '/' + img+'.jpg')
    print('')

    # print(list(matched_data.items())[:1])

    for c, imgs in matched_data.items():
        len_val = len(imgs)//6
        random.shuffle(imgs)

        # build training data
        dst_path = 'data/train/'+c
        os.makedirs(dst_path, exist_ok=True)
        for img in imgs[len_val:]:
            shutil.copy(img, dst_path)

        # build validate data
        dst_path = 'data/val/'+c
        os.makedirs(dst_path, exist_ok=True)
        for img in imgs[:len_val]:
            shutil.copy(img, dst_path)

