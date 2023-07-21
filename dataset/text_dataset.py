import os
import json
import random
from torch.utils.data import Dataset
import h5py

class TextDataset(Dataset):
    """HDF5 dataset"""
    def __init__(self, input_file, roberta_tokenizer, iter_per_epoch=50000000): #default iter > image-text pair instance
        # self.input_file = '/dev/shm/'+input_file
        self.input_file = '/tmp/'+input_file
        self.sent_to_doc = json.load(open(self.input_file[:-1]+'json'+self.input_file[-1],'r'))['dict']
        self.total_len = len(self.sent_to_doc)
        self.roberta_tokenizer = roberta_tokenizer
        self.iter_per_epoch = iter_per_epoch

    def __len__(self):
        return self.iter_per_epoch

    def __getitem__(self, index):
        with h5py.File(self.input_file, 'r') as all_documents:
            document_index = str(self.sent_to_doc[random.randint(0, self.total_len - 1)])
            document = all_documents[document_index]
            i = random.randint(0, len(document) - 1)
            current_chunk = []
            current_length = 0
            while i < len(document):
                segment = document[i]
                current_chunk.extend(segment)
                current_length += len(segment)
                if current_length >= 600:
                    break
                i += 1
            text = self.roberta_tokenizer.decode(current_chunk)
            # text = '[CLS] ' + text + ' [SEP]'
            text = text + ' [SEP]'
        return text