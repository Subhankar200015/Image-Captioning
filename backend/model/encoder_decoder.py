import torchvision.models as models
import torch.nn as nn
import torch

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove final FC layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, 256)  # Project features to embedding size

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 2048)
        features = self.embed(features)  # (batch_size, 256)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])  # skip <end> token
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)  # prepend image features
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs