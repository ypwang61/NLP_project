#Package pre-requisites: openai, pdf2image, tqdm
from openai import OpenAI
import random
import subprocess
import os, shutil, stat, errno
from tqdm import tqdm
from pdf2image import convert_from_path

random.seed(0)
topics =["Education", "Finance", "Health", "Technology", "Science", "Business", "Politics", "History", "Art", "Culture", "Sports", "Entertainment", "Food", "Travel", "Fashion", "Music", "Movies", "Books", "TV Shows", "Video Games", "Social Media", "Internet", "Space", "Environment", "Climate Change", "Sustainability", "Human Rights", "Equality", "Diversity", "Inclusion", "Mental Health", "Physical Health", "Nutrition", "Fitness", "Wellness", "Parenting", "Relationships", "Self-Improvement", "Productivity", "Motivation", "Inspiration", "Creativity", "Innovation", "Entrepreneurship", "Leadership", "Management", "Marketing", "Sales", "Customer Service", "Human Resources",]
random.shuffle(topics)
length_constraint = [5, 4, 3]
num_topics = 25
topics = topics[:num_topics]
slides_per_topic = 5

def generate_text(topic):
    client = OpenAI(api_key='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')
    n_keypoints = random.randint(1, 3)
    prompt = f"Your job is to generate a brief PPT presentation outline on {topic}. Your generation should contain exactly one slide, with {n_keypoints} keypoints. \
    Each key point should be concise, followed with a paragraph demonstrating the corresponding details, which should contain at most {length_constraint[n_keypoints - 1]} sentences. \
    Each key point, together with the demonstrative paragraph, should span one line. \
    There should be exactly one empty line in your output seperating different keypoints. \
    Your generation should follow exactly this format: \nSlide title: [your title]\n"

    for i in range(n_keypoints):
        prompt += f"Keypoint {i+1}: [your keypoint]\nDetails: [your paragraph]\n\n"
    #print(prompt)

    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    return response, n_keypoints

def parse_response(response, n_keypoints):
    success = True
    title, keypoints, details = "", [], []
    lines = response.splitlines()
    if lines[0].startswith("Slide title:"):
        title = lines[0].split(":", 1)[1].strip() 
    else:
        success = False
    for line in lines[1:]:
        if not line:
            continue
        elif line.startswith("Keypoint"):
            keypoints.append(line.split(":", 1)[1].strip())
        elif line.startswith("Details"):
            details.append(line.split(":", 1)[1].strip())
        else:
            success = False
    if len(keypoints) != len(details) or len(keypoints) != n_keypoints:
        success = False
    return title, keypoints, details, success

def write_latex(title, keypoints, details, n_keypoints, topic, index):
    themes = ["default", "Darmstadt", "Malmoe", "AnnArbor", "Dresden", "Marburg", "Antibes", "Frankfurt", "Montpellier","Bergen", "Goettingen", "PaloAlto", "Berkeley", "Hannover", "Pittsburgh", "Berlin", "Ilmenau", "Rochester", "Boadilla", "JuanLesPins", "Singapore", "CambridgeUS", "Luebeck", "Szeged", "Copenhagen", "Madrid"]
    color_themes = ["albatross", "beaver", "beetle", "crane", "default", "dolphin", "dove", "fly", "lily", "orchid", "rose", "seagull", "seahorse", "spruce", "whale", "wolverine"]
    block_width = [0.7, 0.45, 0.3]
    with open(f"Dataset_Generation/Latex/{topic}_{index}.tex", "w+") as f:
        f.write("\\documentclass[5pt]{beamer}\n")
        f.write("\\usetheme{" + random.choice(themes) + "}\n")
        f.write("\\usecolortheme{" + random.choice(color_themes) + "}\n")
        f.write("\\setbeamerfont{block title}{size=\\small}\n")
        f.write("\\setbeamerfont{block body}{size=\\scriptsize}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{frame}\n")
        f.write("\\frametitle{" + title + "}\n")
        f.write("\\begin{columns}\n")
        for i in range(n_keypoints):
            f.write("\\begin{column}{"+ str(block_width[n_keypoints - 1]) +"\\textwidth}\n")
            f.write("\\begin{block}{\\textbf{" + keypoints[i].replace("&", r"\&") +"}}\n")
            f.write(details[i].replace("&", r"\&") + "\n")
            f.write("\\end{block}\n")
            f.write("\\end{column}\n")
        f.write("\\end{columns}\n")
        f.write("\\end{frame}\n")
        f.write("\\end{document}\n")
    
def latex_2_image(topic, index):
    subprocess.run([
        "pdflatex", "-interaction=batchmode",
        "--output-directory=Dataset_Generation/Latex/aux_files",
        f"Dataset_Generation/Latex/{topic}_{index}.tex"
        ])
    os.rename(f"Dataset_Generation/Latex/aux_files/{topic}_{index}.pdf",
              f"Dataset_Generation/pdfs/{topic}_{index}.pdf")
    images = convert_from_path(f"Dataset_Generation/pdfs/{topic}_{index}.pdf")
    images[0].save(f"Dataset_Generation/images/{topic}_{index}.png", "PNG")

def write_keypoints(topic, index, title, keypoints):
    with open(f"Dataset_Generation/keypoints/{topic}_{index}.txt", "w+") as f:
        f.write(topic + "\n")
        f.write(title + "\n")
        for keypoint in keypoints:
            f.write(keypoint + "\n")

def handleRemoveReadonly(func, path, exc):
    '''Handle read-only files for Windows OS.'''
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
        func(path)
    else:
        raise

if __name__ == "__main__":
    if os.path.exists("Dataset_Generation"):
        shutil.rmtree("Dataset_Generation", ignore_errors=False, onerror=handleRemoveReadonly)
    os.mkdir("Dataset_Generation")
    os.mkdir("Dataset_Generation/Latex")
    os.mkdir("Dataset_Generation/Latex/aux_files")
    os.mkdir("Dataset_Generation/pdfs")
    os.mkdir("Dataset_Generation/images")
    os.mkdir("Dataset_Generation/keypoints")

    for j in tqdm(range(num_topics)):
        topic = topics[j]
        for i in range(slides_per_topic):
            response, n_keypoints = generate_text(topic)
            title, keypoints, details, success = parse_response(response, n_keypoints)
            if not success:
                print(f"Failed to parse response for generating {topic} {i+1}/{slides_per_topic} slides")
                continue
            write_latex(title, keypoints, details, n_keypoints, topic, i)
            latex_2_image(topic, i)
            write_keypoints(topic, i, title, keypoints)