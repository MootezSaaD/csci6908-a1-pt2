from torch.utils.data import Dataset
import torch

class WordDataset(Dataset):
    def __init__(self, data, vocab_size, genre_size):
        self.data = data
        self.vocab_size = vocab_size
        self.genre_size = genre_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence_indices, target_index, genre_idx = self.data[idx]
        
        genre_one_hot = torch.zeros(self.genre_size, dtype=torch.float32)
        genre_one_hot[genre_idx-1] = 1
        
        return torch.tensor(sequence_indices), genre_one_hot, torch.tensor(target_index)