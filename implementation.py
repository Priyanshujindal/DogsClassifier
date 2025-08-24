import torch
import torch.nn as nn
import torchvision.transforms as transformers
import torchvision.models as models
from PIL import Image
import os
from typing import List, Dict, Tuple
# import matplotlib.pyplot as plt

class DogImageclassifer:
    def __init__(self,model_path:str,device:str="cpu"):
        self.device=torch.device(device if torch.cuda.is_available() else "cpu")
        self.model=None
        self.classes=[]
        self.transform=None

        self._load_model(model_path)
        self._setup_transformers()
    def _load_model(self,model_path:str):
        checkpoints=torch.load(model_path,map_location=self.device)

        self.model=models.efficientnet_b0(pretrained=True)
        num_input=self.model.classifier[1].in_features
        self.model.classifier[1]=nn.Linear(num_input,len(checkpoints['classes_list']))

        self.classes=checkpoints['classes_list']

        self.model.load_state_dict(checkpoints['model'])
        

        self.model.eval()
        self.model.to(self.device)

    def _setup_transformers(self):
        self.transform=transformers.Compose([
            transformers.Resize((224,224)),
            transformers.ToTensor(),
        ])

    def predict(self,image):
        try:
            self.model.eval()
            img=image.convert('RGB')

            image_tensor=self.transform(img).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                output=self.model(image_tensor)
                probabilities=torch.softmax(output,dim=1)
            
            breed_idx=torch.argmax(probabilities,dim=1).item()
            confidence=probabilities[0][breed_idx].item()
            breed=self.classes[breed_idx]

            # Try a very low threshold - your model has extremely low confidence
            if confidence < 0.05:  # 5% threshold
                return 0,0

            return breed, confidence * 100  # Convert to percentage
        except Exception as e:
            print(f"Prediction error: {e}")
            return F"Prediction error: {e}"
def predict_dog_breed(image,model_path:str='checkpoint_Efficent_B0_classes.pth'):
    classifier=DogImageclassifer(model_path)
    prediction=classifier.predict(image)

    return prediction

if __name__=='__main__':
    print("Dog Image Classifer -Ready to use")




            


