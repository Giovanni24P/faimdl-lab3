import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

class DataLoader():
     """
     Applies some transformations to the data and gives you a DataLoader object
     """
     def __init__(self, root:str, transformations:T.Compose = None):
          if transformations is None:
               self.transform = T.Compose([
                    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ])
          else:
               self.transform = transformations
          
          self.root = root
          self.imageFolder = ImageFolder(root, transformations = self.transform)


     def getCardinality(self) -> int:
          return len(self.imageFolder)

     def getDataLoader(self, batch_size:int = 32, shuffle:bool = True, num_workers:int=8) -> torch.utils.data.DataLoader:
          return torch.utils.data.DataLoader(self.imageFolder, batch_size = batch_size, shuffle = shuffle, num_workers=num_workers)