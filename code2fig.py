
import re
from api_utils import *


def get_query_list_of_flowchart_dataset(prompt_list):
    step1_prompt = ''.join(open('./instruction/generate_dataset_step_1.txt', encoding='utf-8').readlines())
    step2_prompt = ''.join(open('./instruction/generate_dataset_step_2.txt', encoding='utf-8').readlines())
    query_list = []
    for prompt in prompt_list:
        query = [step1_prompt + prompt, step2_prompt]
        query_list.append(query)
    return query_list


def get_processed_res_of_flowchart_dataset(msg_list):
    code = re.search(r'<code>([\s\S]*)<\\code>', msg_list[1]['content'])
    natural_language = re.search(r'<nl>([\s\S]*)<\\nl>', msg_list[3]['content'])
    return {
        'code': code.group(1).strip() if code is not None else "",
        'nl': natural_language.group(1).strip() if natural_language is not None else ""
    }


if __name__ == '__main__':
    prompt_list = ["Education", "Finance", "Health", "Technology", "Science", "Business", "Politics", "History", "Art", "Culture", "Sports", "Entertainment", "Food", "Travel", "Fashion", "Music", "Movies", "Books", "TV Shows", "Video Games"]

    # Generate codes and nl by GPT-4
    res_list = batch_query_openai(prompt_list, get_query_list_of_flowchart_dataset, get_processed_res_of_flowchart_dataset)
    for i in range(len(res_list)):
        with open('./code/{}.txt'.format(i), 'w') as f:
            f.write(res_list[i]['code'])
        with open('./natural language/{}.txt'.format(i), 'w') as f:
            f.write(res_list[i]['nl'])

    # TODO: manually generate original images.

    # Reshape the original images to white-background square
    for i in range(len(prompt_list)):
        make_image_square_with_white_background('./origin/{}.png'.format(i))

    # TODO: manually reconstruct the original images.
    # TODO: manually check the correctness of the OCR results
