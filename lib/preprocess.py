import os, cv2
import sys
import numpy as np
import random
import json
random.seed(1234)
np.random.seed(1234)

def process_head(image, lms, target_size):
    image_height, image_width, _ = image.shape
    lms_x = np.array(lms[0::2])
    lms_y = np.array(lms[1::2])
    lms_x = lms_x / image_width
    lms_y = lms_y / image_height
    lms = np.concatenate([lms_x.reshape(-1,1), lms_y.reshape(-1,1)], axis=1).flatten()
    lms = list(lms)
    image_crop = cv2.resize(image, (target_size[1], target_size[0]), interpolation = cv2.INTER_AREA)
    return image_crop, lms

def process_head_new(image, lms, target_size):
    image_height, image_width, _ = image.shape
    lms_x = np.array(lms[0::2])
    lms_y = np.array(lms[1::2])
    lms_x = lms_x / image_width
    lms_y = lms_y / image_height
    lms = np.concatenate([lms_x.reshape(-1,1), lms_y.reshape(-1,1)], axis=1).flatten()
    lms = list(lms)
    image_crop = cv2.resize(image, (target_size[1], target_size[0]), interpolation = cv2.INTER_AREA)
    return image_crop, lms

def process_jsrt(image, lms, target_size):
    image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    lms = np.array(lms[:188])/1024.
    lms = lms.tolist()
    return image, lms

def process_mont(image, lms_r, lms_l, target_size):
    image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    lms_r = (lms_r.flatten()/1024.).tolist()
    lms_l = (lms_l.flatten()/1024.).tolist()
    lms = lms_r+lms_l
    return image, lms

def process_sz(image, lms_r, lms_l, target_size):
    image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    lms_r = (lms_r.flatten()/1024.).tolist()
    lms_l = (lms_l.flatten()/1024.).tolist()
    lms = lms_r+lms_l
    return image, lms

def process_pad(image, lms, target_size):
    image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    lms = np.array(lms[:188])/1024.
    lms = lms.tolist()
    return image, lms

def gen_data(root_folder, data_name, target_size):
    if not os.path.exists(os.path.join(root_folder, data_name)):
        os.mkdir(os.path.join(root_folder, data_name))
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_train')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_train'))
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_test')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_test'))
    ############################################################################
    if data_name == 'Head':
        # training data
        image_folder = '../data/Head/RawImage'
        anno1_folder = '../data/Head/AnnotationsByMD/400_junior'
        anno2_folder = '../data/Head/AnnotationsByMD/400_senior'

        image_names = sorted(os.listdir(image_folder))
        image_names_train = image_names[:150]
        image_names_test = image_names[150:]

        with open('../data/Head/train.txt', 'w') as f:
            for image_name in image_names_train:
                print(image_name)
                image = cv2.imread(os.path.join(image_folder, image_name))
                anno_name = image_name[:-3]+'txt'

                with open(os.path.join(anno1_folder, anno_name), 'r') as ff:
                    anno1 = ff.readlines()
                anno1 = anno1[:19]
                anno1 = [x.strip().split(',') for x in anno1]
                anno1 = [x for l in anno1 for x in l]
                anno1 = [int(x) for x in anno1]

                with open(os.path.join(anno2_folder, anno_name), 'r') as ff:
                    anno2 = ff.readlines()
                anno2 = anno2[:19]
                anno2 = [x.strip().split(',') for x in anno2]
                anno2 = [x for l in anno2 for x in l]
                anno2 = [int(x) for x in anno2]

                anno_avg = [(x1+x2)/2. for x1, x2 in zip(anno1, anno2)]

                image_crop, lms = process_head(image, anno_avg, target_size)
                cv2.imwrite(os.path.join('../data/Head/images_train', image_name), image_crop)
                f.write(image_name+' ')
                for l in lms:
                    f.write(str(l)+' ')
                f.write('\n')

        # test data
        with open('../data/Head/test.txt', 'w') as f:
            for image_name in image_names_test:
                print(image_name)
                image = cv2.imread(os.path.join(image_folder, image_name))
                anno_name = image_name[:-3]+'txt'

                with open(os.path.join(anno1_folder, anno_name), 'r') as ff:
                    anno1 = ff.readlines()
                anno1 = anno1[:19]
                anno1 = [x.strip().split(',') for x in anno1]
                anno1 = [x for l in anno1 for x in l]
                anno1 = [int(x) for x in anno1]

                with open(os.path.join(anno2_folder, anno_name), 'r') as ff:
                    anno2 = ff.readlines()
                anno2 = anno2[:19]
                anno2 = [x.strip().split(',') for x in anno2]
                anno2 = [x for l in anno2 for x in l]
                anno2 = [int(x) for x in anno2]

                anno_avg = [(x1+x2)/2. for x1, x2 in zip(anno1, anno2)]

                image_crop, lms = process_head(image, anno_avg, target_size)
                cv2.imwrite(os.path.join('../data/Head/images_test', image_name), image_crop)
                f.write(image_name+' ')
                for l in lms:
                    f.write(str(l)+' ')
                f.write('\n')
    elif data_name == 'HeadNew':
        image_folder = '../data/HeadNew/Cephalograms'
        anno1_folder = '../data/HeadNew/Cephalometric_Landmarks/Junior_Orthodontists'
        anno2_folder = '../data/HeadNew/Cephalometric_Landmarks/Senior_Orthodontists'

        processed_all = {}

        image_names = sorted(os.listdir(image_folder))
        for image_name in image_names:
            print(image_name)

            points = {'Sella': [], # 1
                      'Nasion': [], # 2
                      'Orbitale': [], # 3
                      'Porion': [], # 4
                      'A-point': [], # 5
                      'B-point': [], # 6
                      'Pogonion': [], # 7
                      'Menton': [], # 8
                      'Gnathion': [], # 9
                      'Gonion': [], # 10
                      'Lower Incisor Tip': [], # 11
                      'Upper Incisor Tip': [], # 12
                      'Labrale superius': [], # 13
                      'Labrale inferius': [], # 14
                      'Subnasale': [], # 15
                      'Soft Tissue Pogonion': [], # 16
                      'Posterior Nasal Spine': [], # 17
                      'Anterior Nasal Spine': [], # 18
                      'Articulare': [] # 19
                      }

            anno_name = image_name.split('.')[0]+'.json'
            image = cv2.imread(os.path.join(image_folder, image_name))
            with open(os.path.join(anno1_folder, anno_name), 'r') as ff:
                anno1 = json.load(ff)
            with open(os.path.join(anno2_folder, anno_name), 'r') as ff:
                anno2 = json.load(ff)

            for a1, a2 in zip(anno1['landmarks'], anno2['landmarks']):
                assert a1['title'] == a2['title']
                if a1['title'] in points and len(points[a1['title']])==0:
                    x1 = a1['value']['x']
                    y1 = a1['value']['y']
                    x2 = a2['value']['x']
                    y2 = a2['value']['y']
                    x = round((x1+x2)/2)
                    y = round((y1+y2)/2)
                    points[a1['title']].append(x)
                    points[a1['title']].append(y)
            points_list = []
            for k in points:
                points_list += points[k]

            image_crop, lms = process_head_new(image, points_list, target_size)
            processed_all[image_name] = [image_crop, lms]

        with open('../data/HeadNew/train_list.txt', 'r') as f:
            train_list = f.readlines()
        train_list = [x.strip() for x in train_list]
        with open('../data/HeadNew/test_list.txt', 'r') as f:
            test_list = f.readlines()
        test_list = [x.strip() for x in test_list]

        processed_train = [processed_all[x] for x in train_list]
        processed_test = [processed_all[x] for x in test_list]

        with open('../data/HeadNew/train.txt', 'w') as f:
            for item in processed_train:
                image_name = item[0]
                image_crop = item[1][0]
                lms = item[1][1]
                cv2.imwrite(os.path.join('../data/HeadNew/images_train', image_name), image_crop)
                f.write(image_name+' ')
                for l in lms:
                    f.write(str(l)+' ')
                f.write('\n')

        with open('../data/HeadNew/test.txt', 'w') as f:
            for item in processed_test:
                image_name = item[0]
                image_crop = item[1][0]
                lms = item[1][1]
                cv2.imwrite(os.path.join('../data/HeadNew/images_test', image_name), image_crop)
                f.write(image_name+' ')
                for l in lms:
                    f.write(str(l)+' ')
                f.write('\n')
    elif data_name == 'JSRT':
        # JSRT
        image_folder_origin = '../data/JSRT/Images'
        anno_folder_origin = '../data/JSRT/annos'
        
        image_folder_train = '../data/JSRT/images_train'
        if not os.path.exists(image_folder_train):
            os.mkdir(image_folder_train)
        image_names = sorted(os.listdir(image_folder_origin))
        annos_new = []
        with open('../data/JSRT/train.txt', 'w') as f:
            for image_name in image_names:
                print(image_name)
                image = cv2.imread(os.path.join(image_folder_origin, image_name))
                anno_name = image_name[:-3]+'npy'
                anno = np.load(os.path.join(anno_folder_origin, anno_name))
                image, lms = process_jsrt(image, anno, target_size)
                cv2.imwrite(os.path.join(image_folder_train, image_name), image)
        
                f.write(image_name+' ')
                for l in lms:
                    f.write(str(l)+' ')
                f.write('\n')
        os.system('rmdir {}'.format(os.path.join(root_folder, data_name, 'images_test')))
    elif data_name == 'MSP':
        # Montgomery
        print('Processing Montgomery')
        train_ratio = 0.7
        image_folder_origin = '../data/Montgomery/Images'
        anno_RL_folder_origin = '../data/Montgomery/annos_RL'
        anno_LL_folder_origin = '../data/Montgomery/annos_LL'
        
        image_folder_train = '../data/Montgomery/images_train'
        image_folder_test = '../data/Montgomery/images_test'
        if not os.path.exists(image_folder_train):
            os.mkdir(image_folder_train)
        if not os.path.exists(image_folder_test):
            os.mkdir(image_folder_test)
        image_names = sorted(os.listdir(image_folder_origin))
        processed_all = {}
        for image_name in image_names:
            #print(image_name)
            image = cv2.imread(os.path.join(image_folder_origin, image_name))
            anno_name = image_name[:-3]+'npy'
            anno_RL = np.load(os.path.join(anno_RL_folder_origin, anno_name))
            anno_LL = np.load(os.path.join(anno_LL_folder_origin, anno_name))
            image, lms = process_mont(image, anno_RL, anno_LL, target_size)
            processed_all[image_name] = [image, lms]

        with open('../data/Montgomery/train_list.txt', 'r') as f:
            train_list = f.readlines()
        train_list = [x.strip() for x in train_list]
        with open('../data/Montgomery/test_list.txt', 'r') as f:
            test_list = f.readlines()
        test_list = [x.strip() for x in test_list]

        processed_train = [processed_all[x] for x in train_list]
        processed_test = [processed_all[x] for x in test_list]
        
        with open('../data/Montgomery/train.txt', 'w') as f:
            for item in processed_train:
                image_name = item[0]
                image = item[1][0]
                anno = item[1][1]
                cv2.imwrite(os.path.join(image_folder_train, image_name), image)
                f.write(image_name+' ')
                for a in anno:
                    f.write(str(a)+' ')
                f.write('\n')
        
        with open('../data/Montgomery/test.txt', 'w') as f:
            for item in processed_test:
                image_name = item[0]
                image = item[1][0]
                anno = item[1][1]
                cv2.imwrite(os.path.join(image_folder_test, image_name), image)
                f.write(image_name+' ')
                for a in anno:
                    f.write(str(a)+' ')
                f.write('\n')
        #######################################################
        # Shenzhen
        print('Processing Shenzhen')
        train_ratio = 0.7
        image_folder_origin = '../data/Shenzhen/Images'
        anno_RL_folder_origin = '../data/Shenzhen/annos_RL'
        anno_LL_folder_origin = '../data/Shenzhen/annos_LL'
        
        image_folder_train = '../data/Shenzhen/images_train'
        image_folder_test = '../data/Shenzhen/images_test'
        if not os.path.exists(image_folder_train):
            os.mkdir(image_folder_train)
        if not os.path.exists(image_folder_test):
            os.mkdir(image_folder_test)
        image_names = sorted(os.listdir(image_folder_origin))
        processed_all = {}
        for image_name in image_names:
            #print(image_name)
            image = cv2.imread(os.path.join(image_folder_origin, image_name))
            anno_name = image_name[:-3]+'npy'
            anno_RL = np.load(os.path.join(anno_RL_folder_origin, anno_name))
            anno_LL = np.load(os.path.join(anno_LL_folder_origin, anno_name))
            image, lms = process_sz(image, anno_RL, anno_LL, target_size)
            processed_all[image_name] = [image, lms]

        with open('../data/Shenzhen/train_list.txt', 'r') as f:
            train_list = f.readlines()
        train_list = [x.strip() for x in train_list]
        with open('../data/Shenzhen/test_list.txt', 'r') as f:
            test_list = f.readlines()
        test_list = [x.strip() for x in test_list]

        processed_train = [processed_all[x] for x in train_list]
        processed_test = [processed_all[x] for x in test_list]
        
        with open('../data/Shenzhen/train.txt', 'w') as f:
            for item in processed_train:
                image_name = item[0]
                image = item[1][0]
                anno = item[1][1]
                cv2.imwrite(os.path.join(image_folder_train, image_name), image)
                f.write(image_name+' ')
                for a in anno:
                    f.write(str(a)+' ')
                f.write('\n')
        
        with open('../data/Shenzhen/test.txt', 'w') as f:
            for item in processed_test:
                image_name = item[0]
                image = item[1][0]
                anno = item[1][1]
                cv2.imwrite(os.path.join(image_folder_test, image_name), image)
                f.write(image_name+' ')
                for a in anno:
                    f.write(str(a)+' ')
                f.write('\n')
        #######################################################
        # Padchest
        print('Processing Padchest')
        train_ratio = 0.7
        image_folder_origin = '../data/Padchest/Images'
        anno_folder_origin = '../data/Padchest/annos'
        
        image_folder_train = '../data/Padchest/images_train'
        image_folder_test = '../data/Padchest/images_test'
        if not os.path.exists(image_folder_train):
            os.mkdir(image_folder_train)
        if not os.path.exists(image_folder_test):
            os.mkdir(image_folder_test)
        image_names = sorted(os.listdir(image_folder_origin))
        processed_all = {}
        for image_name in image_names:
            #print(image_name)
            image = cv2.imread(os.path.join(image_folder_origin, image_name))
            anno_name = image_name[:-3]+'npy'
            anno = np.load(os.path.join(anno_folder_origin, anno_name))
            image, lms = process_pad(image, anno, target_size)
            processed_all[image_name] = [image, lms]

        with open('../data/Padchest/train_list.txt', 'r') as f:
            train_list = f.readlines()
        train_list = [x.strip() for x in train_list]
        with open('../data/Padchest/test_list.txt', 'r') as f:
            test_list = f.readlines()
        test_list = [x.strip() for x in test_list]

        processed_train = [processed_all[x] for x in train_list]
        processed_test = [processed_all[x] for x in test_list]
        
        with open('../data/Padchest/train.txt', 'w') as f:
            for item in processed_train:
                image_name = item[0]
                image = item[1][0]
                anno = item[1][1]
                cv2.imwrite(os.path.join(image_folder_train, image_name), image)
                f.write(image_name+' ')
                for a in anno:
                    f.write(str(a)+' ')
                f.write('\n')
        
        with open('../data/Padchest/test.txt', 'w') as f:
            for item in processed_test:
                image_name = item[0]
                image = item[1][0]
                anno = item[1][1]
                cv2.imwrite(os.path.join(image_folder_test, image_name), image)
                f.write(image_name+' ')
                for a in anno:
                    f.write(str(a)+' ')
                f.write('\n')
        #######################################################
        # merge
        print('Generating MSP set by merging Montgomery, Shenzhen, and Padchest')
        data_list = ['Montgomery', 'Shenzhen', 'Padchest']
        image_folder_train = '../data/MSP/images_train'
        image_folder_test = '../data/MSP/images_test'
        
        for data in data_list:
            os.system('cp ../data/{}/images_train/* ../data/MSP/images_train/.'.format(data))
            os.system('cp ../data/{}/images_test/* ../data/MSP/images_test/.'.format(data))
            with open('../data/{}/train.txt'.format(data), 'r') as f:
                annos = f.read()
            with open('../data/MSP/train.txt', 'a') as f:
                f.write(annos)
            with open('../data/{}/test.txt'.format(data), 'r') as f:
                annos = f.read()
            with open('../data/MSP/test.txt', 'a') as f:
                f.write(annos)
    else:
        print('Wrong data!')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input one of the data names:')
        print('1. Head')
        print('2. HeadNew')
        print('3. JSRT')
        print('4. MSP')
        exit(0)
    else:
        data_name = sys.argv[1]
        # h x w
        if data_name == 'Head':
            gen_data('../data', data_name, (800, 640))
        elif data_name == 'HeadNew':
            gen_data('../data', data_name, (800, 640))
        elif data_name == 'JSRT':
            gen_data('../data', data_name, (512, 512))
        elif data_name == 'MSP':
            gen_data('../data', data_name, (512, 512))
