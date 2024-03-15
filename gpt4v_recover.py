import argparse
import random
import re
import os
from api_utils import *


def get_processed_res_of_recovery(msg_list):
    rec_txt = re.search(r'<recognized text>([\s\S]*)<\\recognized text>', msg_list[1]['content'])
    cor_txt_wo_context = re.search(r'<corrected text>([\s\S]*)<\\corrected text>', msg_list[1]['content'])
    cor_txt_w_context = re.search(r'<corrected text>([\s\S]*)<\\corrected text>', msg_list[3]['content'])
    return {
        'rec_txt': rec_txt.group(1).strip() if rec_txt is not None else "",
        'cor_txt_wo_context': cor_txt_wo_context.group(1).strip() if rec_txt is not None else "",
        'cor_txt_w_context': cor_txt_w_context.group(1).strip() if rec_txt is not None else "",
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--ori_red_circles_dir', type=str, default='origin_red_circles',
                        help='the directory containing the original red circles')
    parser.add_argument('--red_circles_dir', type=str, default='red_circles',
                        help='the directory containing the red circles')
    parser.add_argument('--texts_dir', type=str, default='code', help='the directory containing the texts')
    parser.add_argument('--output_dir', type=str, default='recover result',
                        help='the directory containing the evaluation results')
    parser.add_argument('--strengths', type=list, default=['50%', '70%'],
                        help='list of strengths to evaluate')
    parser.add_argument('--sample_num', type=int, default='3',
                        help='number of samples for each recovery result')

    args = parser.parse_args()
    random.seed(114514)
    config = vars(args)
    strengths = config['strengths']
    sample_num = config['sample_num']
    ori_red_circles_dir = config['ori_red_circles_dir']
    red_circles_dir = config['red_circles_dir']
    texts_dir = config['texts_dir']
    output_dir = config['output_dir']

    # sample sample_num red circles
    pic_names = os.listdir(ori_red_circles_dir)
    instruction_step1 = ''.join(open('./instruction/recover_step_1.txt', encoding='utf-8').readlines())
    instruction_step2 = ''.join(open('./instruction/recover_step_2.txt', encoding='utf-8').readlines())
    for pic_name in pic_names:
        circle_idxs: list[str] = os.listdir(os.path.join(ori_red_circles_dir, pic_name))
        circle_idxs = [circle_idx for circle_idx in circle_idxs if circle_idx.split('.')[1] == 'jpg' and circle_idx != '_all.jpg']
        circle_idxs = random.sample(circle_idxs, sample_num)
        for strength in strengths:
            pic_strength_dir = os.path.join(red_circles_dir, strength, pic_name)
            for circle_idx in circle_idxs:
                pseudocode = ''.join(open('./{}/{}.txt'.format(texts_dir, pic_name), encoding='utf-8').readlines())
                prompt_list = [
                    form_mm_content(instruction_step1, os.path.join(pic_strength_dir, circle_idx)),
                    instruction_step2 + pseudocode
                ]
                _, res_list = query_openai(prompt_list, 0, get_processed_res_of_recovery, model="gpt-4-vision-preview", max_tokens=500)

                # create output file paths
                output_path_direct_rec = os.path.join(output_dir, 'direct recognize', strength, pic_name)
                os.makedirs(output_path_direct_rec, exist_ok=True)
                output_path_w_context = os.path.join(output_dir, 'with context', strength, pic_name)
                os.makedirs(output_path_w_context, exist_ok=True)
                output_path_wo_context = os.path.join(output_dir, 'without context', strength, pic_name)
                os.makedirs(output_path_wo_context, exist_ok=True)
                with open(output_path_direct_rec + '/' + circle_idx.split('.')[0] + '.txt', 'w', encoding='utf-8') as f:
                    f.write(res_list['rec_txt'])
                with open(output_path_wo_context + '/' + circle_idx.split('.')[0] + '.txt', 'w', encoding='utf-8') as f:
                    f.write(res_list['cor_txt_wo_context'])
                with open(output_path_w_context + '/' + circle_idx.split('.')[0] + '.txt', 'w', encoding='utf-8') as f:
                    f.write(res_list['cor_txt_w_context'])



