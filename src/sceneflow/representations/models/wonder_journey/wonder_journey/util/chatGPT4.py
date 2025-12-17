import openai
import json
import time
from pathlib import Path
import io
import base64
import requests
import spacy
import os
# run 'python -m spacy download en_core_web_sm' to load english language model
nlp = spacy.load("en_core_web_sm")

openai.api_key = os.environ['OPENAI_API_KEY']

class TextpromptGen(object):
    
    def __init__(self, root_path, control=False):
        super(TextpromptGen, self).__init__()
        self.model = "gpt-4" 
        self.save_prompt = True
        self.scene_num = 0
        if control:
            self.base_content = "Please generate scene description based on the given information:"
        else:
            self.base_content = "Please generate next scene based on the given scene/scenes information:"
        self.content = self.base_content
        self.root_path = root_path

    def write_json(self, output, save_dir=None):
        if save_dir is None:
            save_dir = Path(self.root_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            output['background'][0] = self.generate_keywords(output['background'][0])
            with open(save_dir / 'scene_{}.json'.format(str(self.scene_num).zfill(2)), "w") as json_file:
                json.dump(output, json_file, indent=4)
        except Exception as e:
            pass
        return
    
    def write_all_content(self, save_dir=None):
        if save_dir is None:
            save_dir = Path(self.root_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'all_content.txt', "w") as f:
            f.write(self.content)
        return
    
    def regenerate_background(self, style, entities, scene_name, background=None):
        
        if background is not None:
            content = "Please generate a brief scene background with Scene name: " + scene_name + "; Background: " + str(background).strip(".") + ". Entities: " + str(entities) + "; Style: " + str(style)
        else:
            content = "Please generate a brief scene background with Scene name: " + scene_name + "; Entities: " + str(entities) + "; Style: " + str(style)

        messages = [{"role": "system", "content": "You are an intelligent scene generator. Given a scene and there are 3 most significant common entities. please generate a brief background prompt about 50 words describing common things in the scene. You should not mention the entities in the background prompt. If needed, you can make reasonable guesses."}, \
                    {"role": "user", "content": content}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            timeout=5,
        )
        background = response['choices'][0]['message']['content']

        return background.strip(".")
    
    def run_conversation(self, style=None, entities=None, scene_name=None, background=None, control_text=None):

        ######################################
        # Input ------------------------------
        # scene_name: str
        # entities: List(str) ['entity_1', 'entity_2', 'entity_3']
        # style: str
        ######################################
        # Output -----------------------------
        # output: dict {'scene_name': [''], 'entities': ['', '', ''], 'background': ['']}

        if control_text is not None:
            self.scene_num += 1
            scene_content = "\n{Scene information: " + str(control_text).strip(".") + "; Style: " + str(style) + "}"
            self.content = self.base_content + scene_content
        elif style is not None and entities is not None:
            assert not (background is None and scene_name is None), 'At least one of the background and scene_name should not be None'

            self.scene_num += 1
            if background is not None:
                if isinstance(background, list):
                    background = background[0]
                scene_content = "\nScene " + str(self.scene_num) + ": " + "{Background: " + str(background).strip(".") + ". Entities: " + str(entities) + "; Style: " + str(style) + "}"
            else:
                if isinstance(scene_name, list):
                    scene_name = scene_name[0]
                scene_content = "\nScene " + str(self.scene_num) + ": " + "{Scene name: " + str(scene_name).strip(".") + "; Entities: " + str(entities) + "; Style: " + str(style) + "}"
            self.content += scene_content
        else:
            assert self.scene_num > 0, 'To regenerate the scene description, you should have at least one scene content as prompt.'
        
        if control_text is not None:
            messages = [{"role": "system", "content": "You are an intelligent scene description generator. Given a sentence describing a scene, please translate it into English if not and summarize the scene name and 3 most significant common entities in the scene. You also have to generate a brief background prompt about 50 words describing the scene. You should not mention the entities in the background prompt. If needed, you can make reasonable guesses. Please use the format below: (the output should be json format)\n \
                        {'scene_name': ['scene_name'], 'entities': ['entity_1', 'entity_2', 'entity_3'], 'background': ['background prompt']}"}, \
                        {"role": "user", "content": self.content}]
        else:
            messages = [{"role": "system", "content": "You are an intelligent scene generator. Imaging you are flying through a scene or a sequence of scenes, and there are 3 most significant common entities in each scene. Please tell me what sequentially next scene would you likely to see? You need to generate the scene name and the 3 most common entities in the scene. The scenes are sequentially interconnected, and the entities within the scenes are adapted to match and fit with the scenes. You also have to generate a brief background prompt about 50 words describing the scene. You should not mention the entities in the background prompt. If needed, you can make reasonable guesses. Please use the format below: (the output should be json format)\n \
                        {'scene_name': ['scene_name'], 'entities': ['entity_1', 'entity_2', 'entity_3'], 'background': ['background prompt']}"}, \
                        {"role": "user", "content": self.content}]
            
        for i in range(10):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    timeout=5,
                )
                response = response['choices'][0]['message']['content']
                try:
                    print(response)
                    output = eval(response)
                    _, _, _ = output['scene_name'], output['entities'], output['background']
                    if isinstance(output, tuple):
                        output = output[0]
                    if isinstance(output['scene_name'], str):
                        output['scene_name'] = [output['scene_name']]
                    if isinstance(output['entities'], str):
                        output['entities'] = [output['entities']]
                    if isinstance(output['background'], str):
                        output['background'] = [output['background']]
                    break
                except Exception as e:
                    assistant_message = {"role": "assistant", "content": response}
                    user_message = {"role": "user", "content": "The output is not json format, please try again:\n" + self.content}
                    messages.append(assistant_message)
                    messages.append(user_message)
                    print("An error occurred when transfering the output of chatGPT into a dict, chatGPT4, let's try again!", str(e))
                    continue
            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                print("Wait for a second and ask chatGPT4 again!")
                time.sleep(1)
                continue
        
        if self.save_prompt:
            self.write_json(output)

        return output

    def generate_keywords(self, text):
        doc = nlp(text)

        adj = False
        noun = False
        text = ""
        for token in doc:
            if token.pos_ != "NOUN" and token.pos_ != "ADJ":
                continue
            
            if token.pos_ == "NOUN":
                if adj:
                    text += (" " + token.text)
                    adj = False
                    noun = True
                else:
                    if noun:
                        text += (", " + token.text)
                    else:
                        text += token.text
                        noun = True
            elif token.pos_ == "ADJ":
                if adj:
                    text += (" " + token.text)
                else:
                    if noun:
                        text += (", " + token.text)
                        noun = False
                        adj = True
                    else:
                        text += token.text
                        adj = True

        return text

    def generate_prompt(self, style, entities, background=None, scene_name=None):
        assert not (background is None and scene_name is None), 'At least one of the background and scene_name should not be None'
        if background is not None:
            if isinstance(background, list):
                background = background[0]
                
            background = self.generate_keywords(background)
            prompt_text = "Style: " + style + ". Entities: "
            for i, entity in enumerate(entities):
                if i == 0:
                    prompt_text += entity
                else:
                    prompt_text += (", " + entity)
            prompt_text += (". Background: " + background)
            print('PROMPT TEXT: ', prompt_text)
        else:
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
            prompt_text = "Style: " + style + ". " + scene_name + " with " 
            for i, entity in enumerate(entities):
                if i == 0:
                    prompt_text += entity
                elif i == len(entities) - 1:
                    prompt_text += (", and " + entity)
                else:
                    prompt_text += (", " + entity)

        return prompt_text

    def encode_image_pil(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def evaluate_image(self, image, eval_blur=True):
        api_key = openai.api_key
        base64_image = self.encode_image_pil(image)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": ""
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
            "max_tokens": 300
        }

        border_text = "Along the four borders of this image, is there anything that looks like thin border, thin stripe, photograph border, painting border, or painting frame? Please look very closely to the four edges and try hard, because the borders are very slim and you may easily overlook them. I would lose my job if there is a border and you overlook it. If you are not sure, then please say yes."
        print(border_text)
        has_border = True
        payload['messages'][0]['content'][0]['text'] = border_text + " Your answer should be simply 'Yes' or 'No'."
        for i in range(5):
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=5)
                border = response.json()['choices'][0]['message']['content'].strip(' ').strip('.').lower()
                if border in ['yes', 'no']:
                    print('Border: ', border)
                    has_border = border == 'yes'
                    break
            except Exception as e:
                print("Something has been wrong while asking GPT4V. Wait for a second and ask chatGPT4 again!")
                time.sleep(1)
                continue

        if eval_blur:
            blur_text = "Does this image have a significant blur issue or blurry effect caused by out of focus around the image edges? You only have to pay attention to the four borders of the image."
            print(blur_text)
            payload['messages'][0]['content'][0]['text'] = blur_text + " Your answer should be simply 'Yes' or 'No'."
            for i in range(5):
                try:
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=5)
                    blur = response.json()['choices'][0]['message']['content'].strip(' ').strip('.').lower()
                    if blur in ['yes', 'no']:
                        print('Blur: ', blur)
                        break
                except Exception as e:
                    print("Something has been wrong while asking GPT4V. Wait for a second and ask chatGPT4 again!")
                    time.sleep(1)
                    continue
            has_blur = blur == 'yes'
        else:
            has_blur = False

        openai.api_key = api_key
        return has_border, has_blur