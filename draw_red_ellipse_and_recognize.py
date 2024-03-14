# need to install mmocr first: (ref: https://github.com/open-mmlab/mmocr)

# conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
# conda activate open-mmlab
# pip3 install openmim
# git clone https://github.com/open-mmlab/mmocr.git
# cd mmocr
# mim install -e .


# The main function of this script is to draw red ellipses on the bounding boxes of the text detected by the mmocr model
# {
#     'predictions' : [
#       # Each instance corresponds to an input image
#       {
#         'det_polygons': [...],  # 2d list of length (N,), format: [x1, y1, x2, y2, ...]
#         'det_scores': [...],  # float list of length (N,)
#         'det_bboxes': [...],   # 2d list of shape (N, 4), format: [min_x, min_y, max_x, max_y]
#         'rec_texts': [...],  # str list of length (N,)
#         'rec_scores': [...],  # float list of length (N,)
#         'kie_labels': [...],  # node labels, length (N, )
#         'kie_scores': [...],  # node scores, length (N, )
#         'kie_edge_scores': [...],  # edge scores, shape (N, N)
#         'kie_edge_labels': [...]  # edge labels, shape (N, N)
#       },
#       ...
#     ],
#     'visualization' : [
#       array(..., dtype=uint8),
#     ]
# }


import argparse
import cv2
import numpy as np
from mmocr.apis import TextDetInferencer, MMOCRInferencer
import os
import re

def draw_ellipses_all(image_path, ellipse_info_list, output_directory, w = None, h = None):
    """
    draw epplipses on the bounding boxes and store the pictures with all the ellipses
    
    """
    image = cv2.imread(image_path)
    if w is not None and h is not None:
        image = cv2.resize(image, (w, h))
        
    for ellipse_info in ellipse_info_list:
        center, axes = ellipse_info
        cv2.ellipse(image, center, axes, 0, 0, 360, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_directory, '_all.jpg'), image)

def draw_ellipses_each(image_path, ellipse_info_list, rec_texts, output_directory, store_rec_texts = True, w = None, h = None):
    """
    draw epplipses on the bounding boxes and store the pictures with each ellipse
    
    """
        
    for i in range(len(ellipse_info_list)):
        # need to read the image again
        image = cv2.imread(image_path)
        if w is not None and h is not None:
            image = cv2.resize(image, (w, h))
            
        center, axes = ellipse_info_list[i]
    
        cv2.ellipse(image, center, axes, 0, 0, 360, (0, 0, 255), 2)
        
        # use regular expression to remove the special characters that is invalid for the file name, like \
        text = re.sub(r'[\\/*?:"<>|]', '', rec_texts[i])
        
        cv2.imwrite(os.path.join(output_directory, f'{i}_{text}.jpg'), image) # for convenience
        cv2.imwrite(os.path.join(output_directory, f'{i}.jpg'), image)
        
        # store the original text in i.txt
        if store_rec_texts:
            with open(os.path.join(output_directory, f'{i}.txt'), 'w') as f:
                f.write(rec_texts[i])


def group_ij(ellipse_info_list, rec_texts, i, j, scale, test_scale_x, test_scale_y):
    """
    roup the neibor polygons together. For two (center, axes) pairs, if |c1-c2| < |a1| + |a2|, then they are neibors and should be grouped together
    
    """
    c1, a1 = ellipse_info_list[i]
    c2, a2 = ellipse_info_list[j]
    
    if abs(c1[0] - c2[0]) <= (a1[0] + a2[0]) * (test_scale_x / scale) and abs(c1[1] - c2[1]) <= (a1[1] + a2[1])* (test_scale_y / scale):
        # group them together and obtain the new center and axes
        minx, maxx = min(c1[0]-a1[0], c2[0]-a2[0]), max(c1[0]+a1[0], c2[0]+a2[0])
        miny, maxy = min(c1[1]-a1[1], c2[1]-a2[1]), max(c1[1]+a1[1], c2[1]+a2[1])
        
        new_axes = ((maxx - minx) // 2, (maxy - miny) // 2)
        new_center = (minx + new_axes[0], miny + new_axes[1])
        
        # update the ellipse_info_list
        ellipse_info_list[i] = (new_center, new_axes)
        ellipse_info_list.pop(j)
        
        # update the rec_texts in order,from let to right, from top to bottom
        if c1[1] < c2[1] - a2[1] * (0.5 / scale):
            rec_texts[i] = rec_texts[i] + ' ' + rec_texts[j]
        elif c1[1] > c2[1] + a2[1] * (0.5 / scale):
            rec_texts[i] = rec_texts[j] + ' ' + rec_texts[i]
        else:
            if c1[0] < c2[0]:
                rec_texts[i] = rec_texts[i] + ' ' + rec_texts[j]
            else:
                rec_texts[i] = rec_texts[j] + ' ' + rec_texts[i]
        
        rec_texts.pop(j)
        
        # print(f'******* len(ellipse_info_list) = {len(ellipse_info_list)}')
        return True
    
    return False


def reindex_by_order(ellipse_info_list, rec_texts):
    """
    reindex the ellipse_info_list and rec_texts by the order of the center of the ellipses, first compare y, then compare x
    """
    index_sort = sorted(range(len(ellipse_info_list)), key=lambda k: (ellipse_info_list[k][0][1], ellipse_info_list[k][0][0]))
    ellipse_info_list = [ellipse_info_list[i] for i in index_sort]
    rec_texts = [rec_texts[i] for i in index_sort]
    
    # print(f' sorted ellipse_info_list: {ellipse_info_list}')
    # print(f' sorted rec_texts: {rec_texts}')
        
    return ellipse_info_list, rec_texts

def filter_small_bbox(image, ellipse_info_list, rec_texts, min_area = 90):
    """
    filter the small bounding boxes
    """
    new_ellipse_info_list = []
    new_rec_texts = []
    for i in range(len(ellipse_info_list)):
        center, axes = ellipse_info_list[i]
        if axes[0] * axes[1] > min_area:
            new_ellipse_info_list.append(ellipse_info_list[i])
            new_rec_texts.append(rec_texts[i])
    
    return new_ellipse_info_list, new_rec_texts


def polygons_preprocess(results, min_area, group_neibor = False, scale = 1.5, test_scale_x = 1.3, test_scale_y = 1.4):
    """
    Args:
    results: dict, the result of the TextDetInferencer
    group_neibor: bool, if True, group the neibor polygons together
    
    scale: scale to draw the ellipses
    test_scale: scale to test if two polygons are neibors
    
    return ellipse_info_list: list of (center, axes), where center is a tuple of (x, y), axes is a tuple of (a, b)
    """
    
    polygons = results['predictions'][0]['det_polygons']
    for i in range(len(polygons)):
        polygons[i] = np.array(polygons[i], dtype=np.float32).reshape(-1, 2)
    
    ellipse_info_list = []
    # get center, axes first
    for polygon in polygons:
        bbox = cv2.boundingRect(polygon)
        center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
        axes = (int(bbox[2] / 2 * scale), int(bbox[3] / 2 * scale))
        ellipse_info_list.append((center, axes))
    
    # print(f' initial ellipse_info_list: {ellipse_info_list}')
    print(f' len(ellipse_info_list) = {len(ellipse_info_list)}')
    
    rec_texts = results['predictions'][0]['rec_texts']
    # rec is matched with the polygons, so the length of rec_texts should be the same as the length of ellipse_info_list
    
    if group_neibor:
        
        ellipse_info_list, rec_texts = reindex_by_order(ellipse_info_list, rec_texts)
        
        # this is not a efficient way to implement the group_neibor, but just for simplicity
        while 1:
            # loop until each polygon in the ellipse_info_list cannot be grouped with any other polygon
            is_merge = False
            
            # filter the small bounding boxes
            ellipse_info_list, rec_texts = filter_small_bbox(image, ellipse_info_list, rec_texts, min_area)
            
            # reindex because some elements are merged, and their relationship is changed
            ellipse_info_list, rec_texts = reindex_by_order(ellipse_info_list, rec_texts)
            
            for i in range(len(ellipse_info_list)):
                # print(f'################  len(ellipse_info_list) = {len(ellipse_info_list)}')
                for j in range(i+1, len(ellipse_info_list)):
                    
                    is_merge = group_ij(ellipse_info_list, rec_texts, i, j, scale, test_scale_x, test_scale_y)
                    if is_merge:
                        break
                if is_merge:
                    break
            
            if not is_merge:
                break
        
            
    
    return ellipse_info_list, rec_texts
        
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw red ellipses on bounding boxes.')
    parser.add_argument('--device', type=str, help='device to use', default = 'cuda')
    parser.add_argument('--input', type=str, help='Path to the input directory contained the images need to draw ellipses', default = 'ori/')
    parser.add_argument('--output', type=str, help='Path to the output directory, will create directory for each input picture in this path', default = 'ori_red_circles/')
    parser.add_argument('--add_input', type=str, help='Path to the additional input directory, should have the same red circle as input does', default = 'reconstructions/')
    parser.add_argument('--add_output', type=str, help='to store the results from add_input', default = 'red_circles/')
    
    parser.add_argument('--test_scale_x', type=float, help='scale x axes to test if two polygons are neibors', default = 1.3)
    parser.add_argument('--test_scale_y', type=float, help='scale y axes to test if two polygons are neibors', default = 1.4)
    parser.add_argument('--scale', type=float, help='scale to draw the ellipses', default = 1.5)
    parser.add_argument('--min_filter_area_rate', type=float, help='min rate of the total size of image to filter the small bounding boxes', default = 0.0002)
    
    args = parser.parse_args()
    # print the parameters of args in lines
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
        
    # inferencer = TextDetInferencer(model='DBNet', device=args.device)
    # recognize the text in the image
    inferencer = MMOCRInferencer(det='DBNET', rec='SAR', device=args.device)
    
    
    ############## read all the images in the input directory and draw ellipses on them ##############
    for image_name in os.listdir(args.input):
        print(f' ================== begin add ellipses to {image_name} ===============')
        image_path = os.path.join(args.input, image_name)
        results = inferencer(image_path, return_vis=False)
        # dict_keys(['rec_texts', 'rec_scores', 'det_polygons', 'det_scores'])
        
        rec_texts = results['predictions'][0]['rec_texts']
        
        # obatain the size of the image
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        min_area = h * w * args.min_filter_area_rate
        print(f' h = {h}, w = {w}, min_area = {min_area}')
        
        
        ellipse_info_list, rec_texts = polygons_preprocess(results, min_area, group_neibor = True, scale = args.scale, test_scale_x = args.test_scale_x, test_scale_y = args.test_scale_y)
        
        print(f"Final rec_texts = {rec_texts}")
        print(f"Final ellipse_info_list = {ellipse_info_list}, len(ellipse_info_list) = {len(ellipse_info_list)}")
        # create a directory for each image
        directory = os.path.join(args.output, image_name.split('.')[0])
        
        flag = 0
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            ###### draw each ellipse and store the pictures in a directory for each image ######
            draw_ellipses_each(image_path, ellipse_info_list, rec_texts, directory)
            
            ###### draw all the ellipses and store the pictures in a directory for each image to see the ocr detection results ######
            draw_ellipses_all(image_path, ellipse_info_list, directory)
        else:
            print(f' {directory} already exists, so skip this image.')
            flag = 1
        
        
        ######### read the additional input directory and draw ellipses on them #########
        # first find the file_name that contain image_name
        
        if flag == 0:
            for add_image_name in os.listdir(args.add_input):
                if image_name.split('.')[0] in add_image_name : # 
                    print(f' ================== begin add ellipses to {add_image_name} ===============')
                    
                    
                    add_image_path = os.path.join(args.add_input, add_image_name)
                    add_output_path = os.path.join(args.add_output, add_image_name.split('.')[0])
                    if not os.path.exists(add_output_path):
                        os.makedirs(add_output_path)
                    
                        # print the size of the image
                        add_image = cv2.imread(add_image_path)
                        h_new, w_new, _ = add_image.shape
                        print(f' h_new = {h_new}, w_new = {w_new}')
                        
                        # resize the add_image to the same size as the image
                        add_image = cv2.resize(add_image, (w, h))
                        print(f'add_image.shape = {add_image.shape}')
                        
                        # just draw the ellipses on the add_image_path with the original ellipse_info_list
                        draw_ellipses_each(add_image_path, ellipse_info_list, rec_texts, add_output_path, store_rec_texts = False, w = w, h = h)
                        
                        draw_ellipses_all(add_image_path, ellipse_info_list, add_output_path, w = w, h = h)
                    else:
                        print(f' {add_output_path} already exists, so skip this image.')