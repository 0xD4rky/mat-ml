# utils/train.py
import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, num_epochs=50, lr=0.001):
    # Loss and Optimizer
    criterion = nn.MSELoss()  # Example loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, timesteps in train_loader:
            images = images.unsqueeze(1)  # Add channel dimension
            outputs = model(images)
            
            # Example: MSE between model output and timestep (dummy task for illustration)
            loss = criterion(outputs.squeeze(), timesteps.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    print("Training completed!")
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
