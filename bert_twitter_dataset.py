from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        length = torch.sum(attention_mask).item()

        return {
            'tweet_text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'length': length,
            'labels': torch.tensor(label).float()
        }

# Example usage
#dataset = SentimentDataset(["Sample text"], [1])
#loader = DataLoader(dataset, batch_size=1)
