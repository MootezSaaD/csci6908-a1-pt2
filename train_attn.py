import random
random.seed(4224)
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch
import json, os, pickle, random
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from model2.utils import sample_n_items_from_pickle, read_json_as_dict, train_epoch, test_epoch
from model2.model import LSTMNextWordPredictor
from model2.dataset import WordDataset
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


data_root = "./data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train set is huge (~6.1M), Im taking a sample out of it.
# Same thing for dev and test that contain around 100k each.
TRAIN_SAMLPE = int(100e4)
DEV_SAMPLE, TEST_SAMPLE = int(10e3), int(10e3)

train = sample_n_items_from_pickle(os.path.join(data_root, "train_2.pkl"), TRAIN_SAMLPE)
test = sample_n_items_from_pickle(os.path.join(data_root, "test_2.pkl"), TEST_SAMPLE)
dev = sample_n_items_from_pickle(os.path.join(data_root, "dev_2.pkl"), DEV_SAMPLE)

VOCAB_SIZE = int(5e3)


genres2idx = read_json_as_dict("./data/genres2idx.json")


batch_size = 512
embedding_dim = 64
hidden_dim = 128
num_layers = 2
dropout = 0.3
num_epochs = 350
adam_epsilon = 1e-8
learning_rate = 0.000005
weight_decay = 1e-4
genre_size = len(genres2idx)

train_dataset = WordDataset(train, VOCAB_SIZE, genre_size)
test_dataset = WordDataset(test, VOCAB_SIZE, genre_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True)

model = LSTMNextWordPredictor(
    vocab_size=VOCAB_SIZE+1, 
    embedding_dim=embedding_dim, 
    hidden_dim=hidden_dim, 
    grene_size=genre_size, 
    num_layers=num_layers, 
    dropout=dropout)

max_steps = num_epochs*len(train_loader)

model.to(device)


# Define the loss function
loss_fn =  nn.CrossEntropyLoss()

# Define the optimizer
no_decay = ["bias"]

optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters(
    ) if not any(nd in n for nd in no_decay)]}
]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=learning_rate, eps=adam_epsilon, weight_decay = weight_decay)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.01,
                                            num_training_steps=max_steps)


epoch_progress_bar = tqdm(range(num_epochs), desc='Epochs')

model.zero_grad()
for epoch in epoch_progress_bar:
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
    test_loss = test_epoch(model, test_loader, loss_fn, device)
    
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
          f'Test Loss: {test_loss:.4f}')




writer.flush()
writer.close()
torch.save(model.state_dict(), './model2_state_dict.pth')
