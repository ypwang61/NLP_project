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

## 1. Automatically draw red circles on the original images and reconstruction images

Just run 
```
python draw_red_ellipse_and_recognize.py
```
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
    - 📊 1.png
    - ...
- 📂 red_circles/
  - 📂 p0_70per/
    - 📊 0.jpg
    - 📊 1.jpg
    - ...

## 2. Apply GPT-4V for correcting the content in the red circles

```
python gpt4v_recover.py --dropout_circle_idx_list 0 2 --debug 0
```
Here `dropout_circle_idx_list` denotes the pictures in `/red_circles/xxx/` that you don't want to pay attention to, and debug=1 will not use GPT-4V API.

This script will output the correction content and the stats values (edit distance, GLEU, semantic similarity using Bert), and store in `evaluation_results`.
