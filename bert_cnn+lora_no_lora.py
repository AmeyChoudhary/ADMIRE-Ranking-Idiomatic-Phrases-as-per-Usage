# %%
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
import numpy as np 
import pandas as pd 
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt

# %%
import pandas as pd

# Load the TSV file to check its structure
file_path = 'Data/train_subtaskA/subtask_a_train.tsv'
df = pd.read_csv(file_path, sep='\t')

# Display the first few rows to understand the structure
df.head()

# %%
import ast
def arrange_sentences(df):
    captions_ordered = []   
    
    for i in range(len(df)):
        order = df['expected_order'][i]
        image1_name = df['image1_name'].values[i]
        image2_name = df['image2_name'].values[i]
        image3_name = df['image3_name'].values[i]
        image4_name = df['image4_name'].values[i]
        image5_name = df['image5_name'].values[i]
        
        captions = []
        order = ast.literal_eval(order)
        for j in range(len(order)):
            image_name = order[j]
            if image_name == image1_name:
                captions.append(df['image1_caption'].values[i])
            elif image_name == image2_name:
                captions.append(df['image2_caption'].values[i])
            elif image_name == image3_name:
                captions.append(df['image3_caption'].values[i])
            elif image_name == image4_name:
                captions.append(df['image4_caption'].values[i])
            elif image_name == image5_name:
                captions.append(df['image5_caption'].values[i])
        captions_ordered.append(captions)

    return captions_ordered

captions_ordered = arrange_sentences(df)
df['captions_ordered'] = captions_ordered


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
class AdMireDataset(Dataset):
    def __init__(self, df, max_length=512):
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        compound = self.df.iloc[idx]['compound']
        sentence = self.df.iloc[idx]['sentence']
        captions = [
            self.df.iloc[idx]['image1_caption'],
            self.df.iloc[idx]['image2_caption'],
            self.df.iloc[idx]['image3_caption'],
            self.df.iloc[idx]['image4_caption'],
            self.df.iloc[idx]['image5_caption']
        ]

        caption1 = self.df.iloc[idx]['image1_caption']
        caption2 = self.df.iloc[idx]['image2_caption']
        caption3 = self.df.iloc[idx]['image3_caption']
        caption4 = self.df.iloc[idx]['image4_caption']
        caption5 = self.df.iloc[idx]['image5_caption']
        expected_order = self.df.iloc[idx]['captions_ordered']

        # expected_order contains sentences ordered, replace it with indices
        expected_order_indices = []
        for i in range(len(expected_order)):
            s = expected_order[i]
            if s == caption1:
                expected_order_indices.append(1)
            elif s == caption2:
                expected_order_indices.append(2)
            elif s == caption3:
                expected_order_indices.append(3)
            elif s == caption4:
                expected_order_indices.append(4)
            elif s == caption5:
                expected_order_indices.append(5)

        # Replace the compound in sentence and captions with [SEP] markers
        sentence = sentence.replace(compound, '[SEP] ' + compound + ' [SEP]')
        captions = [cap.replace(compound, '[SEP] ' + compound + ' [SEP]') for cap in captions]

        # Tokenize sentence and captions
        sentence_tokens = self.tokenizer(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        caption_tokens = [self.tokenizer(cap, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt') for cap in captions]

        # Find the token position of the compound (marked by [SEP] compound [SEP])
        # Compute compound position and convert it to a tensor
        num_compound_words = len(compound.split(' '))
        compound_position = torch.tensor(
            (sentence_tokens['input_ids'][0].tolist().index(self.tokenizer.convert_tokens_to_ids('[SEP]')) + 1 , num_compound_words),
            device=device  # Move to the correct device (GPU or CPU)
        )

        # Convert the list of caption positions to a tensor
        caption_positions = torch.tensor(
            [(cap['input_ids'][0].tolist().index(self.tokenizer.convert_tokens_to_ids('[SEP]')) + 1 , num_compound_words) for cap in caption_tokens],
            device=device  # Make sure the tensor is moved to the correct device (GPU or CPU)
        )


        expected_order_indices = np.array(expected_order_indices)
        expected_order_indices = torch.tensor(expected_order_indices, device = device)

        return sentence_tokens, caption_tokens, expected_order_indices, compound_position, caption_positions


# %%
dataset  = AdMireDataset(df)

# %%
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
# Split the dataset indices
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)

# Create Subsets for training and validation
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)


# %%
dataloader = DataLoader(
    train_dataset,              # The dataset object
    batch_size=1,        # Adjust the batch size as needed
    shuffle=True,         # Shuffle data at each epoch
    collate_fn=None       # We can specify a custom collate_fn if necessary, but for now it's None
)

v_dataloader = DataLoader(
    val_dataset,              # The dataset object
    batch_size=1,        # Adjust the batch size as needed
    shuffle=False,         # Shuffle data at each epoch
    collate_fn=None       # We can specify a custom collate_fn if necessary, but for now it's None
)

# %%
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from transformers import AutoModel

class DualEncoderModelWithLoRA(nn.Module):
    def __init__(self, rank=4):
        super(DualEncoderModelWithLoRA, self).__init__()
        
        # Load the pre-trained BERT model and freeze it
        self.ibert = AutoModelForTokenClassification.from_pretrained("imranraad/idiom-xlm-roberta")
        self.ibert.classifier = nn.Identity()
        self.ibert.requires_grad_(False)  # Freeze all BERT parameters
        
        # Define LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r= rank,               # Rank of the LoRA matrices
            lora_alpha=16,     # Scaling factor
            lora_dropout=0.1,  # Dropout rate for LoRA layers
            target_modules=["query", "value"],  # Apply LoRA to these attention modules
        )
        
        # Apply LoRA to the BERT model
        self.bert1 = get_peft_model(self.ibert, lora_config)
        self.bert2 = get_peft_model(self.ibert, lora_config)
    def forward(self, sentence_tokens, caption_tokens, compound_position, caption_positions):

        # Anchor embedding (sentence compound)
        sentence_tokens = {k: v.squeeze(0) for k, v in sentence_tokens.items()}
        sentence_outputs = self.bert1.base_model.model.roberta(**sentence_tokens)
        # sentence_embedding = sentence_outputs.last_hidden_state[:, compound_position, :]
        # compound_position = compound_position[0]
        sentence_embedding = sentence_outputs.last_hidden_state[compound_position[0][0] : compound_position[0][0] + compound_position[0][1]] # (num_words,768)
        print(sentence_embedding)
        # Caption embeddings (captions compounds)
        caption_embeddings = []
        for i, cap_tokens in enumerate(caption_tokens):
            cap_tokens = {k: v.squeeze(0) for k, v in cap_tokens.items()}
            cap_outputs = self.bert2.base_model.model.roberta(**cap_tokens)
            cap_embedding = cap_outputs.last_hidden_state[caption_positions[0][i][0] : caption_positions[0][i][0] + caption_positions[0][i][1]]
            caption_embeddings.append(cap_embedding)
        
        return sentence_embedding, caption_embeddings


# %%
class EncoderWithBERTCNN(nn.Module):
    def __init__(self, cnn_output_channels=128, cnn_kernel_size=3):
        super(EncoderWithBERTCNN, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.requires_grad_(False)

        # Define CNN layer to process BERT embeddings
        # self.cnn = nn.Conv1d(in_channels=768, out_channels=cnn_output_channels, kernel_size=cnn_kernel_size, padding=1)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, tokens):
        # Process the input using BERT
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        outputs = self.bert(**tokens) 
        embedding = outputs.last_hidden_state # Shape: (batch_size , seq_length, 768)
        
        
        # Apply CNN
        embedding = embedding.permute(0 ,2,1)  # Shape: (1, 768, seq_length)
        cnn_output = self.cnn(embedding).squeeze(0)  # Shape: (cnn_output_channels, seq_length)

        return cnn_output  # Returns the embedding vector of size (128 , seq_length)

class EncoderWithBERTLoRACNN(nn.Module):
    def __init__(self, cnn_output_channels=128, cnn_kernel_size=3):
        super(EncoderWithBERTLoRACNN, self).__init__()
        
        # Load and freeze the pre-trained BERT model
        self.bert_base = AutoModel.from_pretrained("bert-base-uncased")
        for param in self.bert_base.parameters():
            param.requires_grad = False  # Freeze BERT parameters
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r= 4,               # Rank of the LoRA matrices
            lora_alpha=16,     # Scaling factor
            lora_dropout=0.1,  # Dropout rate for LoRA layers
            target_modules=["query", "value"],  # Apply LoRA to these attention modules
        )
        self.bert = get_peft_model(self.bert_base, lora_config)

        # Define CNN layer to process BERT embeddings
        # self.cnn = nn.Conv1d(in_channels=768, out_channels=cnn_output_channels, kernel_size=cnn_kernel_size, padding=1)
        # Let's use a 1D CNN with kernel size 3 and 128 output channels
        # Make multiple CNN layers with different kernel sizes for better performance

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )


    def forward(self, tokens):
        # Process the input using BERT
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        outputs = self.bert(**tokens) 
        embedding = outputs.last_hidden_state # Shape: (batch_size , seq_length, 768)
        
        # Apply CNN
        embedding = embedding.permute(0 ,2,1)  # Shape: (1, 768, seq_length)
        cnn_output = self.cnn(embedding).squeeze(0)  # Shape: (cnn_output_channels, seq_length)

        return cnn_output  # Returns the embedding vector of size (128 , seq_length)
    
class DualEncoderModelWithCNN(nn.Module):
    def __init__(self, cnn_output_channels=128, cnn_kernel_size=3):
        super(DualEncoderModelWithCNN, self).__init__()
        # Instantiate two separate encoders for query and captions
        self.query_encoder = EncoderWithBERTCNN(cnn_output_channels, cnn_kernel_size)
        self.caption_encoder = EncoderWithBERTCNN(cnn_output_channels, cnn_kernel_size)

    def forward(self, query_tokens, caption_tokens, query_positions, caption_positions):
        # Generate embeddings for the query
        query_embedding = self.query_encoder(query_tokens) # (128 , seq_length)
        query_embedding = query_embedding.permute(1,0)[query_positions[0][0] : query_positions[0][0] + query_positions[0][1]] # (num_words, 128)
        
        # Generate embeddings for each caption
        caption_embeddings = []
        for i, cap_tokens in enumerate(caption_tokens):
            cap_embedding = self.caption_encoder(cap_tokens) # (128 , seq_length)
            cap_embedding = cap_embedding.permute(1,0)[caption_positions[0][i][0] : caption_positions[0][i][0] + caption_positions[0][i][1]] # (num_words, 128)
            caption_embeddings.append(cap_embedding)

        return query_embedding, caption_embeddings  # Returns query and list of caption embeddings

# %%
model = DualEncoderModelWithCNN().to(device)

# %%
class RankingTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(RankingTripletLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
    
    def forward(self, sentence_embedding, caption_embeddings, expected_order):
        loss = 0.0
        num_triplets = 0

        expected_order = expected_order[0]
        

        # Iterate over each pair of captions
        for i in range(len(expected_order) - 1):
            for j in range(i + 1, len(expected_order)):
                pos = caption_embeddings[expected_order[i] - 1]
                neg = caption_embeddings[expected_order[j] - 1]
                
                # print(i,j, expected_order,pos.shape , neg.shape , sentence_embedding.shape)
                try:
                    # Calculate pairwise triplet loss
                    d_pos = self.cosine_similarity(sentence_embedding, pos)
                    d_neg = self.cosine_similarity(sentence_embedding, neg)
                    
                    # Weighted loss based on distance in order
                    weight = 1.0 / abs(i - j)
                    triplet_loss = weight * torch.relu(d_neg - d_pos + self.margin)
                    
                    loss += triplet_loss
                    num_triplets += 1
                except:
                    pass

        loss = loss / num_triplets if num_triplets > 0 else 0.0
        return loss.mean()


# %%
# Initialize criterion and optimizer
criterion = RankingTripletLoss(margin=1.0)  # Or any custom loss function
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs= 200
train_loss = []
val_loss_arr = []
best_val_loss = 1e6

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    total_train_loss = 0.0
    num_train_samples = 0
    print("Epoch", epoch + 1)
    
    for batch in dataloader:
        print(num_train_samples, end='\r')
        num_train_samples += 1

        # Unpack batch
        sentence_tokens, caption_tokens, expected_order, compound_positions, caption_positions = batch

        # Move tensors to device
        sentence_tokens = {k: v.to(device) for k, v in sentence_tokens.items()}
        caption_tokens = [{k: v.to(device) for k, v in cap.items()} for cap in caption_tokens]
        expected_order = expected_order.to(device)
        compound_positions = compound_positions.to(device)
        caption_positions = caption_positions.to(device)

        # Forward pass: compute embeddings for the anchor sentence and captions
        sentence_embedding, caption_embeddings = model(sentence_tokens, caption_tokens, compound_positions, caption_positions)
        
        # Compute loss: Ranking Triplet Loss for each batch
        loss = criterion(sentence_embedding, caption_embeddings, expected_order)
        total_train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(dataloader)
    train_loss.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

    # Validation Phase
    model.eval()
    total_val_loss = 0.0
    num_val_samples = 0

    with torch.no_grad():  # Disable gradient calculation for validation
        for batch in v_dataloader:
            num_val_samples += 1

            # Unpack batch
            sentence_tokens, caption_tokens, expected_order, compound_positions, caption_positions = batch

            # Move tensors to device
            sentence_tokens = {k: v.to(device) for k, v in sentence_tokens.items()}
            caption_tokens = [{k: v.to(device) for k, v in cap.items()} for cap in caption_tokens]
            expected_order = expected_order.to(device)
            compound_positions = compound_positions.to(device)
            caption_positions = caption_positions.to(device)

            # Forward pass
            sentence_embedding, caption_embeddings = model(sentence_tokens, caption_tokens, compound_positions, caption_positions)
            
            # Compute loss
            val_loss = criterion(sentence_embedding, caption_embeddings, expected_order)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(v_dataloader)
    val_loss_arr.append(avg_val_loss)

    if (val_loss_arr[-1] < best_val_loss):
        best_val_loss = val_loss_arr[-1]
        torch.save(model.state_dict(), 'best_model_with_cnn_multilayer.pt')
    
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

# %%


plt.plot(train_loss, label='t')
plt.plot(val_loss_arr, label='v')
plt.legend()
plt.savefig('loss_model_with_cnn_multilayer.png')

# %%



