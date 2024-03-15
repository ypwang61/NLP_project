<!--  -->
# NLP Project


The files we need now contain the original pictures (ori/), pictures after applying diffusion models (reconstructions/), captions(texts/) and references images(references/, can be empty). Before running this code, the directory should be setup like
- 📂 ori/
  - 📊 p0.png
- 📂 reconstructions/
  - 📊 p0_70per.png
- 📂 references/
  - 📂 p0/
    - 📊 0.jpg
    - 📊 1.jpg
- 📂 texts/
  - 📄 p0.txt

## 0. Environment

need to install [mmocr](https://github.com/open-mmlab/mmocr).

```
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
mim install -e .
```

And the `bert_score` library
```
pip install evaluate bert_score
```

## 1. Set OpenAI key
```
export NLP_API_KEY='YOUR_API_KEY'
```

## 2. Automatically draw red circles on the original images and reconstruction images

Just run 
```
python draw_red_ellipse_and_recognize.py --random_sample_num 5
```

`random_sample_num` here means that we will randomly select 5 red circle to draw, rather than draw all the red circles.
**NOTE**: Since in this file we first get the ocr bbox of the original pictures, then apply them on the pictures after applying diffusion model, so please make sure that the size and configuration of these two pictures are almost matched. (The function will resize the picture but still need to check the configurations manually.)

After running this script, the files architecture will be like:

- 📂 ori/
  - 📊 p0.png
- 📂 reconstructions/
  - 📊 p0_70per.png
- 📂 references/
  - 📂 p0/
    - 📊 0.jpg
    - 📊 1.jpg
- 📂 texts/
  - 📄 p0.txt
- 📂 ori_red_circles/
  - 📂 p0/
    - 📊 0.png
    - 📄 0.txt
    - 📊 1.png
    - 📄 1.txt
    - ...
- 📂 red_circles/
  - 📂 p0_70per/
    - 📊 0.jpg
    - 📊 1.jpg
    - ...

**NOTE**: You'd better double check the ground truth results in `ori_red_circles/pi/j.txt` since the text recognization model is not always perform perfectly.
**NOTE**: If you want to ignore some red circles, just delete the items in `red_circles` and you don't need to delete anything in the `ori_red_circles` (Although alignment is fine, but it's exhausting).

## 3. Apply GPT-4V for correcting the content in the red circles

When debug, can run as:
```
python gpt4v_recover.py --select_pic_strength_name p0_50per p1_50per p0_70per --debug 1
```
In this format, the program will not call the openai API but use some predefined content to test the program. After making sure that the program works well, you can run

```
python gpt4v_recover.py --debug 0
```
Here `drop_pic_strength_name` denotes the pictures that you don't want to pay attention to, and `select_pic_strength_name` denotes that you just want to try on these examples.

This script will output the correction content and the stats values (edit distance, GLEU, semantic similarity using Bert), and store in `evaluation_results`.


## 4. Recalculate the results
If you run the experiments several times and want to evaluate the average scores among all the results, just run
```
python final_report.py --evaluate_path evaluation_results
```
