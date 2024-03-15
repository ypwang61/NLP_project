import evaluate
import argparse
import os
import pandas as pd


################# evaluation functions #################
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def calculate_edit_distance(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]


def calculate_rouge(text1, text2):
    res = rouge.compute(predictions=[text1], references=[text2])
    return res['rouge1'], res['rouge2'], res['rougeL'], res['rougeLsum']


def calculate_bert_score(text1, text2):
    res = bertscore.compute(predictions=[text1], references=[text2], lang='en')
    return res['precision'][0], res['recall'][0], res['f1'][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--result_dir', type=str, default='recover result',
                        help='the directory containing the GPT-4 recovery results')
    parser.add_argument('--ground_truth_dir', type=str, default='origin_red_circles',
                        help='the directory containing the ground-truth results')
    parser.add_argument('--output_dir', type=str, default='evaluation result',
                        help='the directory to output evaluation csv')

    args = parser.parse_args()
    config = vars(args)
    result_dir = config['result_dir']
    ground_truth_dir = config['ground_truth_dir']
    output_dir = config['output_dir']

    cat_list = os.listdir(result_dir)
    res_pd = pd.DataFrame(columns=['category', 'edit_distance', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bert_score_precision', 'bert_score_recall', 'bert_score_f1'])
    for cat in cat_list:
        sub_cat_list = os.listdir(os.path.join(result_dir, cat))
        for sub_cat in sub_cat_list:
            edit_distance = 0
            rouge1, rouge2, rougeL, rougeLsum = 0.0, 0.0, 0.0, 0.0
            bert_score_precision, bert_score_recall, bert_score_f1 = 0.0, 0.0, 0.0
            count = 0
            full_cat_name = cat + '_' + sub_cat
            image_id_list = os.listdir(os.path.join(result_dir, cat, sub_cat))
            for image_id in image_id_list:
                circle_id_list = os.listdir(os.path.join(result_dir, cat, sub_cat, image_id))
                for circle_id in circle_id_list:
                    prediction = ''.join(open(os.path.join(result_dir, cat, sub_cat, image_id, circle_id), encoding='utf-8').readlines())
                    ground_truth = ''.join(open(os.path.join(ground_truth_dir, image_id, circle_id), encoding='utf-8').readlines())
                    edit_distance += calculate_edit_distance(prediction, ground_truth)
                    rouge_res = calculate_rouge(prediction, ground_truth)
                    rouge1 += rouge_res[0]
                    rouge2 += rouge_res[1]
                    rougeL += rouge_res[2]
                    rougeLsum += rouge_res[3]
                    bert_res = calculate_bert_score(prediction, ground_truth)
                    bert_score_precision += bert_res[0]
                    bert_score_recall += bert_res[1]
                    bert_score_f1 += bert_res[2]
                    count += 1
            res_pd.loc[len(res_pd)] = [full_cat_name, edit_distance / count, rouge1 / count, rouge2 / count, rougeL / count, rougeLsum / count, bert_score_precision / count, bert_score_recall / count, bert_score_f1 / count]
    res_pd.to_csv(os.path.join(output_dir, 'evaluation.csv'), index=False)


