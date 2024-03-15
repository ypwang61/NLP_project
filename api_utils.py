from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import openai
import time
import collections
import io
import base64

api_key = ''


def encode_image(image_path):
    """
    Encode the image into base64 format
    """
    image_resized = Image.open(image_path).resize((1024, 1024))
    buffered = io.BytesIO()
    image_resized.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def form_mm_content(text, image_path):
    content_list = [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(image_path)}",
            },
        }
    ]

    return content_list


def make_image_square_with_white_background(image_path):
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        image = rgb_image

    width, height = image.size
    new_size = max(width, height)
    new_image = Image.new('RGB', (new_size, new_size), (255, 255, 255))

    left = (new_size - width) // 2
    top = (new_size - height) // 2

    new_image.paste(image, (left, top))
    new_image.save(image_path)


def query_openai(prompt_list, index, get_processed_res, model="gpt-4", max_tokens=100, temperature=0.0, n=1):
    msg_list = []
    client = openai.OpenAI(api_key=api_key)
    for prompt_idx in range(len(prompt_list)):
        retry_count = 100
        retry_interval = 1
        msg_list.append({"role": "user", "content": prompt_list[prompt_idx]})
        for _ in range(retry_count):
            try:
                msg = client.chat.completions.create(
                    model=model,
                    messages=msg_list,
                    temperature=temperature,
                    n=n,
                    max_tokens=max_tokens
                )
                msg_list.append({"role": "assistant", "content": msg.choices[0].message.content})
                break
            except Exception as e:
                print("Error info: ", e)
                print('Retrying....')
                retry_count += 1
                retry_interval *= 2
                time.sleep(retry_interval)
    return index, get_processed_res(msg_list)


def batch_query_openai(prompt_list, get_query_list, get_processed_res):
    with ProcessPoolExecutor(max_workers=5) as executor:
        query_list = get_query_list(prompt_list)
        futures = [executor.submit(query_openai, prompt_list, index, get_processed_res) for index, prompt_list in enumerate(query_list)]
        query2res = collections.defaultdict(str)
        for job in as_completed(futures):
            index, res = job.result(timeout=None)
            query2res[index] = res

    return [query2res[i] for i in range(len(prompt_list))]
