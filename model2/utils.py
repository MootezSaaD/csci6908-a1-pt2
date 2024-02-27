import pickle, json, random
from tqdm.auto import tqdm
import torch

def sample_n_items_from_pickle(file_path, n):
    """Samples N items from a list stored in a pickle file.

    Args:
        file_path (str): The path to the pickle file.
        n (int): The number of items to read.

    Returns:
        list: The first N items from the list.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            # Shuflle
            random.shuffle(data)
            # Ensure that the data read is a list
            if not isinstance(data, list):
                raise ValueError(f"Data is a {type(data)}, expected a list.")
            # Return the first N items
            return data[:n]
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
def read_json_as_dict(file_path):
    """
    Reads a JSON file and returns the data as a dictionary.

    Parameters:
    - file_path (str): The file path of the JSON file to read.

    Returns:
    - dict: The dictionary containing the data from the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return None
    
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', position = 0,leave=True)
    model.train()
    for seq, genre , targets in progress_bar:
        seq, genre, targets = seq.to(device), genre.to(device), targets.to(device)
        logits = model(seq, genre)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)



def test_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Testing', position = 0, leave=True)
    with torch.no_grad():
        for seq, genre, targets in progress_bar:
            seq, genre, targets = seq.to(device), genre.to(device), targets.to(device)
            logits = model(seq, genre)
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    return average_loss