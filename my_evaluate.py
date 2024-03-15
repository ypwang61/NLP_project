from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer, BertModel
import torch
from evaluate import load
from functools import partial

EVALUATE_NAMES = [
    'edit_dis', 
    # 'bleu_score', 
    'rouge_score',
    'Bert_sim'
    ]
bertscore = load("bertscore")
rougescore = load("rouge")
################# evaluation functions #################

def calculate_edit_distance(text1, text2):
    # edit distance
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

def calculate_bleu_score(text1, text2):
    # BLEU score
    reference = [text2.split()]
    candidate = text1.split()
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score


def calculate_rouge_score(text1, text2):
    # contain rouge-1, rouge-2, rouge-l, rouge_lsum
    results = rougescore.compute(predictions=[text1], references=[text2])
    # print(f'results = {results}')
    return results


def calculate_semantic_similarity(text1, text2, model_name = "bert-base-uncased", device = "cpu"):
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertModel.from_pretrained(model_name).to(device)
    
    # model.eval()
    
    # inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    # inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
    
    # inputs1 = inputs1.to(device)
    # inputs2 = inputs2.to(device)
    
    # with torch.no_grad():
    #     outputs1 = model(**inputs1)
    #     outputs2 = model(**inputs2)
        
    # embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    # embeddings2 = outputs2.last_hidden_state.mean(dim=1)
    # similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=1).item()
    # return similarity

    
    results = bertscore.compute(predictions=[text1], references=[text2], lang="en", model_type=model_name, device=device)
    
    # results = {'precision': [0.3785819411277771], 'recall': [0.5577176809310913], 'f1': [0.4510134160518646], 'hashcode': 'bert-base-uncased_L9_no-idf_version=0.3.12(hug_trans=4.22.1)'}
    similarity = results['f1'][0]

    return similarity
    






################# evaluation exacution #################



def evaluate_fun(result_list, ocr, ori_ocr_text, device = "cuda"):
    """
    Evaluate the result_list and the ocr text
    
    Args:
    result_list: a list of tuples, each tuple is (guess, reason)
    ocr: the OCR text
    ori_ocr_text: the original OCR text
    
    Returns:
    res_stats_dict: a dictionary containing the evaluation results
    """
    
    ################# evaluation table #################
    

    evaluate_functions = [
    partial(calculate_edit_distance),
    # partial(calculate_bleu_score),
    partial(calculate_rouge_score),
    partial(calculate_semantic_similarity, model_name = "bert-base-uncased", device = device)
    ]

    Higher_is_better = {
    'edit_dis': False,
    # 'bleu_score': True,
    'rouge_score': True,
    'Bert_sim': True
    }

    assert len(EVALUATE_NAMES) == len(evaluate_functions) and len(EVALUATE_NAMES) == len(Higher_is_better)
    
    
    ################# helper functions #################
    def evaluate_helper(text1, text2):
        res = {EVALUATE_NAMES[i] : evaluate_functions[i](text1, text2) 
            for i in range(len(evaluate_functions))}
        # if the element of res is a dictionary(like for rouge), we need to add all the elements in the dictionary as a new key-value pair
        for name in EVALUATE_NAMES:
            if isinstance(res[name], dict):
                for key, value in res[name].items():
                    res[name + '@' + key] = value
                res.pop(name)
        print(f'res = {res}')
        return res



    ################# main function #################
    
    ocr_eval = evaluate_helper(ocr, ori_ocr_text)
    print(f'ocr_eval: {ocr_eval}')
    
    guess_stats_dict = {name: [] for name in EVALUATE_NAMES}
    reason_stats_dict = {name: [] for name in EVALUATE_NAMES}
    
    for guess, reason in result_list:
        res = evaluate_helper(guess, ori_ocr_text)
        rres = evaluate_helper(reason, ori_ocr_text)
        
    # write a form that is not sensitive to the number of evaluation metrics
    # need to consider some key are not inside the EVALUATE_NAMES
        for key in res.keys():
            if key in guess_stats_dict:
                guess_stats_dict[key].append(res[key])
                reason_stats_dict[key].append(rres[key])
            else:
                guess_stats_dict[key] = [res[key]]
                reason_stats_dict[key] = [rres[key]]
                
                
        
    # calculate the avg top_3 scores for each guess and reason
    n = 3
    guess_top_n_values_dict, reason_top_n_values_dict = {}, {}
    for name in res.keys():
        print(f'guess_stats_dict[{name}] = {guess_stats_dict[name] }')
        
        if name in EVALUATE_NAMES:
            higher_is_better = Higher_is_better[name]
        else:
            higher_is_better = Higher_is_better[name.split('@')[0]]
        
        print(f'higher_is_better = {higher_is_better} for {name}')
        if higher_is_better:
            guess_top_n_values_dict[name] = max(guess_stats_dict[name][:n])
            reason_top_n_values_dict[name] = max(reason_stats_dict[name][:n])
        else:
            guess_top_n_values_dict[name] = min(guess_stats_dict[name][:n])
            reason_top_n_values_dict[name] = min(reason_stats_dict[name][:n])
    
    
    # visualize the top_n_values
    
    # print the results
    print(f'==================== pipeline results ====================')
    print(f'ori_ocr_text: {ori_ocr_text}\nocr: {ocr}\nresult_list: {result_list}')
    print(f'==================== evaluation results ====================')
    for name in res.keys():
        print(f'{name}\t OCR: {ocr_eval[name]:.3f}\t top-1 guess: {guess_stats_dict[name][0]:.3f}\t top-3 guess: {guess_top_n_values_dict[name]:.3f}\t \
            top-1 reason: {reason_stats_dict[name][0]:.3f}\t top-3 reason: {reason_top_n_values_dict[name]:.3f}')
    
    
    # store the results in a dictionary so that we can store them in a json file
    res_stats_dict = {
    'ori_ocr_text': ori_ocr_text,
    'ocr': ocr,
    'result_list': result_list,
    }

    for name in res.keys():
        res_stats_dict[name] = {
            'ocr': ocr_eval[name],
            'top_1_guess': guess_stats_dict[name][0],
            'top_3_guess': guess_top_n_values_dict[name],
            'top_1_reason': reason_stats_dict[name][0],
            'top_3_reason': reason_top_n_values_dict[name],
        }
    
    return res_stats_dict, res.keys()


if __name__ == "__main__":
    # example
    text1 = "I have a dream that one day this nation will rise up and live out the true meaning of its creed."
    text2 = "I have a dream that one day this nation will rise up."

    edit_distance = calculate_edit_distance(text1, text2)
    print(f"Edit Distance: {edit_distance}")

    bleu_score = calculate_bleu_score(text1, text2)
    print(f"BLEU Score: {bleu_score:.2f}")

    semantic_similarity = calculate_semantic_similarity(text1, text2)
    print(f"Semantic Similarity: {semantic_similarity:.2f}")