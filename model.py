import torch
import torchvision.models as models
from transformers import BeitForImageClassification

nclasses = 20
class BEiTNet(torch.nn.Module):
    def __init__(self):
        super(BEiTNet, self).__init__()
        self.beit =  BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-224')
        self.beit.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.beit.classifier.in_features,512),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(512,nclasses)
        )

    def forward(self, x):
        return self.beit(x).logits
