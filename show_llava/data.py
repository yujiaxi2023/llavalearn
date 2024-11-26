from typing import Any
from dataclasses import dataclass
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import List

from transformers import AutoProcessor

    
@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values:torch.Tensor
    a_input_ids:torch.Tensor


class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir:str) -> tuple[List, Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images_dl")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        return chat_data, image_dir
    
    def __len__(self):
        return len(self.chat_data)
    
    def __getitem__(self, index):
        cur_data = self.chat_data[index]

        human_input = cur_data['conversations'][0]['value']
        gpt_output = cur_data['conversations'][1]['value']

        image_path = self.image_dir.joinpath(cur_data.get('image'))

        return (human_input, gpt_output, image_path)



def build_qaimage(processor: AutoProcessor, q_text:str, a_text:str, image_path: Path):
    
    # instruction or input or question
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_file = image_path
    raw_image = Image.open(image_file)

    inputs = processor(prompt, raw_image, return_tensors="pt")

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]

    return QaImageOutput(
        q_input_ids=inputs['input_ids'],
        pixel_values=inputs["pixel_values"],
        a_input_ids=a_input_ids
        ) 
 
    
class TrainLLavaModelCollator:
    def __init__(self, processor:AutoProcessor, IGNORE_INDEX:int) -> None:
        self.processor = processor
        self.ignore_index = IGNORE_INDEX

    def convert_one_piece(self,
                          q_input_ids:torch.Tensor,
                          a_input_ids:torch.Tensor):
        input_ids = torch.concat([
            q_input_ids,
            a_input_ids,
            torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], dim=1)
        labels = torch.concat([
            torch.full_like(q_input_ids, fill_value=self.ignore_index),
            a_input_ids,
            torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], dim=1)

        return input_ids, labels
    
    def __call__(self, features:List) -> Any:
        
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            qaimage_output = build_qaimage(
                self.processor,
                feature[0],
                feature[1],
                feature[2]
            )
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids,
                qaimage_output.a_input_ids
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat([
            torch.concat([
                torch.full(
                    (1, max_input_len - max_input_len_list[index]),
                    self.processor.tokenizer.pad_token_id,
                ),value,
            ], axis=1)
            for index, value in enumerate(input_ids_list)
        ])
        
        final_labels = torch.concat([
            torch.concat([
                torch.full(
                    (1, max_input_len - max_input_len_list[index]),
                    self.processor.tokenizer.pad_token_id,
                ),value,
            ], axis=1)
            for index, value in enumerate(labels_list)
        ])

        final_pixel_values = torch.concat(pixel_values, axis=0)
        
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return {
            "input_ids":final_input_ids,
            "labels":final_labels,
            "pixel_values":final_pixel_values,
            "attention_mask":attention_mask
        }

