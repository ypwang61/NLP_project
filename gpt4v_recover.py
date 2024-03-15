

import os
import ipdb

from my_evaluate import evaluate_fun

from api_utils import run_GPT4V_api_one_step, get_result_list_from_content

from final_report import final_report

import json

import argparse


parser = argparse.ArgumentParser(description='')

parser.add_argument('--ori_red_circles_dir', type=str, default='ori_red_circles', help='the directory containing the original red circles')
parser.add_argument('--red_circles_dir', type=str, default='red_circles', help='the directory containing the red circles')
parser.add_argument('--texts_dir', type=str, default='texts', help='the directory containing the texts')
parser.add_argument('--references_dir', type=str, default='references', help='the directory containing the reference images. Can be None if no reference images are available.')
parser.add_argument('--evaluate_results_dir', type=str, default='evaluate_results', help='the directory containing the evaluation results')
parser.add_argument('--red_circle_format', type=str, default='jpg', help='format of the red circle image')

# parser.add_argument('--dropout_circle_idx_list', nargs='+', type=int, default=None, help='the list of red circle indices to process, if None, process all red circles in the red_circles_dir')

parser.add_argument('--select_pic_strength_name', nargs='+', type=str, default=None, help='the list of pic_strength_names to process, if None, process all pic_strength_names in the red_circles_dir')
parser.add_argument('--drop_pic_strength_name', nargs='+', type=str, default=None, help='the list of pic_strength_names not to process, if None, process all pic_strength_names in the red_circles_dir')


parser.add_argument('--device', type=str, default='cuda', help='the device for running the semantic similarity model')
parser.add_argument('--debug', type=int, default=0, help='whether to run in debug mode which does not call the API and use a fixed response instead')


args = parser.parse_args()




def get_paths(config):
  """
  Get the paths for a specific Figure task
  """
  pic_name = config['pic_name']
  strength_postfix = config['strength_postfix']
  circle_idx = config['circle_idx']
  red_circle_format = config['red_circle_format']


  pic_strength_name = f'{pic_name}_{strength_postfix}' # example: p0_70per
  text_path = os.path.join(config['texts_dir'], f'{pic_name}.txt') # caption of the figure, example: texts/p0.txt
  caption_text = open(text_path, 'r').read()
  
  red_circle_recon_image_path = os.path.join(config['red_circles_dir'], pic_strength_name, f'{circle_idx}.{red_circle_format}') # diffusion model image + red circle, the text in the red circle is bad and need to be replaced
  # example: red_circles/p0_70per/10.jpg

  if config['references_dir'] is None or not os.path.exists(config['references_dir']):
    reference_images_path = []
  else:
    reference_dir = os.path.join(config['references_dir'], pic_name) # reference content from paper for particular Figure
    reference_images = os.listdir(reference_dir)
    reference_images_path = [os.path.join(reference_dir, image) for image in reference_images]
    print(f'reference_images_path:')
    for reference_image_path in reference_images_path:
      print(reference_image_path) # example: references/p0/0.png, references/p0/1.png, ...

  ori_ocr_text_path = os.path.join(config['ori_red_circles_dir'], pic_name, f'{circle_idx}.txt') # ocr text in the red circle of the ground truth image
  
  ori_ocr_text = open(ori_ocr_text_path, 'r').read() # example: ori_red_circles/p0/10.txt
  print(f'ori_ocr_text: {ori_ocr_text}')

  evaluate_path = os.path.join(config['evaluate_results_dir'], pic_strength_name, f'{circle_idx}.json') # store the evaluation results, example: evaluate_results/p0_70per/10.json
  if not os.path.exists(os.path.dirname(evaluate_path)):
    os.makedirs(os.path.dirname(evaluate_path))
    
    
  return reference_images_path, red_circle_recon_image_path, evaluate_path, caption_text, ori_ocr_text, 
  
  

def GPT4V_process(config):
  """
  Process one image and the text using GPT-4-V
  
  Args:
    config: a dictionary containing the configuration
    
  Returns:
    content: the response content from the API
    
  
  """
  reference_images_path, red_circle_recon_image_path, evaluate_path, caption_text, ori_ocr_text = get_paths(config)
  
  print(f'================================= get content list from adding prompts, pictures from reference and pictures from red_circle =================================')
  
  if config['debug'] != 0:
    # content = "Step2\n- The framework is designed for joint learning of image resizer and recognition models.\n- The resizer adapts to various models and improves classification performance.\n- It's not limited by pixel or perceptual loss, creating machine adaptive visual effects.\n- Enables down-scaling at arbitrary factors to find optimal resolution for tasks.\n- Extends application to image quality assessment (IQA), performing successfully.\n\nStep3\n1: (Learning, the text refers to the process the model undergoes as it adjusts the resolution and recognizes features in the image.)\n2: (Joint Learning, it references the joint optimization of the resizer and recognition models within the framework.)\n3: (Optimization, it could imply the process of enhancing the image for better compatibility with the recognition model.)\n\nOCR Joint Learning"
    content = "'Step2*** System components: MSE for Mean Squared Error loss, E2E_ASR for End-to-End Automatic Speech Recognition, Fbank for filter bank feature extraction, and a Discriminator for distinguishing features. Enhancement network for noise suppression.\n\nStep3*** 1: (Lgan, common notation for adversarial loss) @ 2: (Lgen, variation of generative adversarial loss) @ 3: (Ladv, abbreviation for adversarial loss). OCR*** Lgui.'"
  else:
    response = run_GPT4V_api_one_step(caption_text, reference_images_path, red_circle_recon_image_path)
    print(response)
    content = response.choices[0].message.content
  
  
  print(f'================================= read content and evaluate =================================')
  
  result_list, ocr = get_result_list_from_content(content)

  print(f'result_list = {result_list}')
  print(f'ocr = {ocr}')

  res_stats_dict = evaluate_fun(result_list, ocr, ori_ocr_text, device=config['device'])
  
  with open(evaluate_path, 'w') as f:
    json.dump(res_stats_dict, f, indent=2)
    print(f'write the evaluation results to {evaluate_path}')
    
  return res_stats_dict


#  res_stats_dict = {
#     'ori_ocr_text': ori_ocr_text,
#     'ocr': ocr,
#     'result_list': result_list,
#     }

#     for name in EVALUATE_NAMES:
#         res_stats_dict[name] = {
#             'ocr': ocr_eval[name],
#             'top_1_guess': guess_stats_dict[name][0],
#             'top_3_guess': guess_top_n_values_dict[name],
#             'top_1_reason': reason_stats_dict[name][0],
#             'top_3_reason': reason_top_n_values_dict[name],
#         }



def main(args):
  # copy args into dictionary config
  config = vars(args)
  
  # first read the dic name in red_circles
  red_circles_dir = config['red_circles_dir']
  pic_strength_names = os.listdir(red_circles_dir)
  
  final_list = {}
  for pic_strength_name in pic_strength_names: # for each Figure task
    
    if config['select_pic_strength_name'] is not None and pic_strength_name not in config['select_pic_strength_name']:
      continue
    
    if config['drop_pic_strength_name'] is not None and pic_strength_name in config['drop_pic_strength_name']:
      continue
    
    
    pic_strength_dir = os.path.join(red_circles_dir, pic_strength_name)
    circle_idxs = os.listdir(pic_strength_dir)
    config['pic_name'] = pic_strength_name.split('_')[0]
    config['strength_postfix'] = pic_strength_name.split('_')[1]
    
    for circle_idx in circle_idxs: # for each red circle in the Figure task
      # contain figures like 0_XXX.jpg and 0.jpg
      circle_idx_str = circle_idx.split('.')[0]
      
      if circle_idx_str.isdigit(): # ignore the file like 0_XXX.jpg
        # if config['dropout_circle_idx_list'] is not None and int(circle_idx_str) in config['dropout_circle_idx_list']:
        #   continue # if the circle_idx is in the dropout list, skip it
        
        config['circle_idx'] = int(circle_idx.split('.')[0])
        print(f'#############################################################################')
        print(f"################### circle_idx = {config['circle_idx']} #####################")
        print(f'#############################################################################')
        print(f'config: {config}')
        
        # process the image and the text using GPT-4-V
        res_stats_dict = GPT4V_process(config)

        final_list[(pic_strength_name,circle_idx) ] = res_stats_dict
        
  final_report(final_list)
  
  # print(f'final_list: {final_list}')
  return final_list





main(args)
ipdb.set_trace()
