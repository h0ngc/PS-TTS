import openai
import json
import os
import argparse
import pdb

parser = argparse.ArgumentParser(description='Text File Translation Script')
parser.add_argument('-i','--input_file', type=str, required=True)
parser.add_argument('-o','--output_file', type=str, required=True)
args = parser.parse_args()


with open('config.json') as f:
    config = json.load(f)

openai.api_key = config['OPENAI_API_KEY']
engine = config['OPENAI_ENGINE']

def translate_text(text, target_language): 
    response = openai.Completion.create( 
        engine=engine, 
        prompt=f"Translate the following text into {target_language} without any additional explanation and write only english : {text}\n", 
        max_tokens=400, 
        n=1, 
        stop=None, 
        temperature=1, 
    ) 
    return response.choices[0].text.strip()


translated_lines = []
segment_lines = []
with open(args.input_file, 'r', encoding='utf-8') as file:
    for line in file:
        number,onset,offset, korean_text = line.strip().split('\t')
        translated_text = translate_text(korean_text, 'english')
        print(translated_text)
        translated_lines.append(f"{number}\t{onset}\t{offset}\t{translated_text}")
with open(args.output_file, 'w', encoding='utf-8') as outfile:
    for line in translated_lines:
        outfile.write(line + '\n')
