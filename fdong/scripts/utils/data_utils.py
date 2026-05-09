import os
import numpy as np
import torch
import json

from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map


class TokenizedJSONLData(Dataset):
    def __init__(self, dataset_dir, max_seq_len, tokenizer, padding=True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        # 递归收集所有子文件夹中的jsonl文件
        self.files = []
        for root, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if filename.endswith('.txt'):
                    self.files.append(os.path.join(root, filename))
        self.files = sorted(self.files)
        # self.files = self.files[0*8800:0*8800+352]+self.files[1*8800:1*8800+210]+self.files[2*8800:2*8800+210]+self.files[3*8800:3*8800+210]
        
        self._load_file(0)  # 加载第一个文件
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.combine_sequence = 1 # int(np.ceil(max_seq_len / 1024))
        self.lines_per_file = len(self.file_texts) // self.combine_sequence
        self.padding = padding

    def __len__(self):
        return len(self.files) * self.lines_per_file

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(self.files[self.cur_file_idx], 'r', encoding='utf-8') as f:
            # 直接读取每行作为文本，去除首尾空白字符
            self.file_texts = [line.strip() for line in f if line.strip()]
            # 过滤掉空行

    def __getitem__(self, index):
        file_idx = index // self.lines_per_file
        if file_idx != self.cur_file_idx:
            self._load_file(file_idx)
        
        # 计算行索引并确保不越界
        line_idx = index % self.lines_per_file
        max_line_idx = len(self.file_texts) // self.combine_sequence - 1
        line_idx = min(line_idx, max_line_idx)

        # 组合多个文本片段
        # start_idx = line_idx * self.combine_sequence
        # end_idx = start_idx + self.combine_sequence
        text = json.loads(self.file_texts[line_idx]) # ''.join(self.file_texts[start_idx:end_idx])

        # 处理tokenization
        if self.padding:
            token_ids = self.tokenizer(
                text, 
                padding='max_length',
                max_length=self.max_seq_len + 1,
                padding_side='right',
                truncation=True,
                return_tensors='pt'
            ).input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = len(token_ids[0])
        
        return token_ids[0, :-1], token_ids[0, 1:], real_len

