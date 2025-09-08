
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader,TensorDataset
import time




class SimpleModel(nn.Module):

    def __init__(self, input_size = 784, hidden_size = 512,num_classes = 10):
        super(SimpleModel,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
    
def creat_dummy_data(batch_size = 32,num_batches = 100):

    # create test data
    X = torch.randn(batch_size*num_batches,784)
    y = torch.randint(0,10,(batch_size*num_batches,))

    # convert to tensor
    dataset = TensorDataset(X,y)
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

    return dataloader


def train_pytorch():
    print("=== Standard PyTorch Training ===")
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"using device: {device}")
    model = SimpleModel().to(device)

    # create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(),lr = 0.001)

    # create dataloader
    dataloader = creat_dummy_data(batch_size=32,num_batches=20)

    # loop
    model.train()
    total_loss = 0
    start_time = time.time()
    for epoch in range(5):
        epoch_loss = 0
        for batch_idx, (data,target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 5 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        print(f'Epoch {epoch} finished, Average Loss: {epoch_loss/len(dataloader):.6f}')
        total_loss += epoch_loss
    end_time = time.time()
    print(f"Training complete in {end_time - start_time:.2f} seconds")
    print(f"Average total loss: {total_loss/(3*len(dataloader)):.6f}")


if __name__ == "__main__":
    train_pytorch()

