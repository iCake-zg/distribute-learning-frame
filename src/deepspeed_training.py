
from torch import nn

import torch
from torch.utils.data import TensorDataset,DataLoader
import argparse
import deepspeed
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


def add_argument():
    
    parser = argparse.ArgumentParser(description="Deepspeed Training Example")

    # training params
    parser.add_argument('--epochs',type=int,default=3,help='number of epochs')
    parser.add_argument('--local_rank',type=int,default=1,help='local rank passed from distributed launcher')

    # Deepspeed training params
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def train_deepspeed():
    """DeepSpeed训练"""
    print("=== DeepSpeed Training ===")
    
    # args
    args = add_argument()

    # model
    model = SimpleModel()

    # data
    dataloader = creat_dummy_data(batch_size=64,num_batches=20)

    #deepspeed initialize
    model_engine,optimizer,_,_ = deepspeed.initialize(
        args = args,
        model = model,
        model_parameters = model.parameters(),
        config='../configs/deepspeed_config.json'
    )

    # loss_function
    criterion = nn.CrossEntropyLoss()

    # loop
    model_engine.train()
    total_loss = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch_idx,(data,target) in enumerate(dataloader):
            data = data.to(model_engine.local_rank)
            target = target.to(model_engine.local_rank)
            output = model_engine(data)
            loss = criterion(output,target)
            model_engine.backward(loss)
            model_engine.step()
            epoch_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        print(f'Epoch {epoch} finished, Average Loss: {epoch_loss/len(dataloader):.6f}')
        total_loss += epoch_loss
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Average total loss: {total_loss/(args.epochs*len(dataloader)):.6f}")
    
    # save_model
    model_engine.save_checkpoint('./checkpoints', tag='final')
    print("Model saved to ./checkpoints")



if __name__ == "__main__":
    train_deepspeed()