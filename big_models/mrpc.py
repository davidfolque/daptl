import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

mrpc = load_dataset('glue', 'mrpc')


model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(mrpc['train']['sentence1'], mrpc['train']['sentence1'], truncation=True, padding='max_length', max_length=256)
val_encodings = tokenizer(mrpc['validation']['sentence1'], mrpc['validation']['sentence2'], truncation=True, padding='max_length', max_length=256)

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TokenizedDataset(train_encodings, mrpc['train']['label'])
val_dataset = TokenizedDataset(val_encodings, mrpc['validation']['label'])

model = BertForSequenceClassification.from_pretrained(model_name)

batch_size = 32

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch.
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size*4,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    save_total_limit=1,              # limit the total amount of checkpoints. Deletes the older checkpoints.    
    fp16 = False,
    fp16_backend = 'apex',
    fp16_opt_level='O3'
)


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
