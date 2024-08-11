import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNForNER(nn.Module):
    def __init__(self, pretrained_model, num_classes, max_length=128):
        super(CNNForNER, self).__init__()
        self.transformer = pretrained_model
        self.max_length = max_length

        # Get the number of labels from the pretrained model
        pretrained_num_labels = self.transformer.num_labels

        self.conv1 = nn.Conv1d(in_channels=pretrained_num_labels, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, pretrained_num_labels)

        # Apply CNN layers
        logits = logits.permute(0, 2, 1)  # Shape: (batch_size, pretrained_num_labels, sequence_length)
        conv1_out = F.relu(self.conv1(logits))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv2_out = self.dropout(conv2_out)
        conv2_out = conv2_out.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, 128)
        final_logits = self.fc(conv2_out)  # Shape: (batch_size, sequence_length, num_classes)
        return final_logits