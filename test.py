from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
from torch import Tensor

def test ():
    
    # Dummy dataset
    X = torch.randn(10000, 100)
    y = torch.randint(0, 2, (10000,))

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )

    for batch_x, batch_y in tqdm(dataloader):
        # batch_x.shape -> [128, 100]
        # batch_y.shape -> [128]
        
        print(batch_x, batch_y)
        
if __name__ == '__main__':
    test()
