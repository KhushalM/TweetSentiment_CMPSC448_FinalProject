from torch.utils.data import Dataset, DataLoader
import torch

class TwitterDataset(Dataset):
    def __init__(self,tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        length = torch.sum(attention_mask).item()

        return {
            'tweet_text': tweet,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'length': length,  # Include sequence length
            'labels': torch.tensor(label, dtype=torch.long)
        }