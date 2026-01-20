from openai import AzureOpenAI
import openai

from gpt_credentials import api_base, api_key, deployment_name, api_version, api_base_uw, api_key_uw

client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

sftclient = AzureOpenAI(
    api_key=api_key_uw,
    api_version=api_version,
    base_url=f"{api_base_uw}/openai/deployments/{deployment_name}"
)

import base64, json
from mimetypes import guess_type

import re
import time

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def call_gpt(prompt, image_path_0 = None, caller_type = None, image_path_1 = None, image_path_2 = None, image_path_3 = None, ):
    # Path to your image
    ms = [
                { "role": "user", 
                 "content": [  
                        { 
                            "type": "text", 
                            "text": prompt
                        },
                        
                        
                    ] 
                } 
            ]
    if image_path_0:
        ms[0]['content'].append({ 
                            "type": "image_url",
                            "image_url": {
                            "url": local_image_to_data_url(image_path_0)
                            }
                        })
    if image_path_1:
        ms[0]['content'].append({ 
                            "type": "image_url",
                            "image_url": {
                            "url": local_image_to_data_url(image_path_1)
                            }
                        })
    if image_path_2:
        ms[0]['content'].append({ 
                            "type": "image_url",
                            "image_url": {
                            "url": local_image_to_data_url(image_path_2)
                            }
                        })
    if image_path_3:
        ms[0]['content'].append({ 
                            "type": "image_url",
                            "image_url": {
                            "url": local_image_to_data_url(image_path_3)
                            }
                        })

    for attempt in range(3):
        try:
            # print(caller_type)
            if caller_type == 'sft':
                response = sftclient.chat.completions.create(
                    model=deployment_name,
                    messages=ms         )
            else:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=ms         )
            response = json.loads(response.json() )
            return response['choices'][0]['message']['content']
        except openai.RateLimitError as e:
            print(e)
            msg = str(e)
            match = re.search(r"retry after (\d+) seconds", msg)
            if match:
                wait = int(match.group(1))
            else:
                wait = 15 * (2 ** attempt)  # fallback exponential backoff
            print(f"Rate limit hit, retrying in {wait} seconds...")
            time.sleep(wait)
        except Exception as e:
            print(e)
            return 'I do not know.'

    return 'I do not know'

