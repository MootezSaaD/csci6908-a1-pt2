import torch.nn as nn
import torch.nn.functional as F
import torch

class LSTMNextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, grene_size, num_layers, dropout):
        super(LSTMNextWordPredictor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.grene_size = grene_size

        # Embedding layer that turns word indices into embeddings
        self.seq_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Linear layer that acts as a embedding layer for a genre
        self.genre_embedding = nn.Linear(grene_size, hidden_dim)
        # Define the LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout = self.dropout_rate, batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)
        # Scale for scaled-dot product attention
        self.scale = 1 / (hidden_dim ** 0.5)
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, seq, genre):
        seq_embd = self.seq_embedding(seq)
        lstm_out, _ = self.lstm(seq_embd)
        genre_embedded = self.genre_embedding(genre.float()).unsqueeze(1)  # (bs, 1, hidden_dim]
        # From slide 88 in Lecture 5
        key = value = lstm_out
        query = genre_embedded
        attn_output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            dropout_p=self.dropout_rate
        )
        attn_output = attn_output.squeeze(1)
        last_hidden_state = lstm_out[:, -1, :]
        combined_output = torch.cat((last_hidden_state, attn_output), dim=1)
        
        output = self.fc(combined_output)
        
        
        return output