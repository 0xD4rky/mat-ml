from loader import *


def get_efficientnet_model(num_classes):
    """
    Returns a pre-trained EfficientNet model with a modified classifier head for classification.
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1  
    model = efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes) 
    return model.to(device)

model = get_efficientnet_model(num_classes=len(class_names))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train(model, dataloader, criterion, optimizer):
    """
    Train the model for one epoch with tqdm progress bar and time tracking.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimizer.

    Returns:
        epoch_loss: Average loss for the epoch.
        epoch_acc: Accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.update(1)

    end_time = time.time()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    print(f"Epoch completed in {end_time - start_time:.2f}s")
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion):
    """
    Evaluate the model on a validation or test dataset with tqdm and time tracking.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader for the validation or test data.
        criterion: Loss function.

    Returns:
        epoch_loss: Average loss for the dataset.
        epoch_acc: Accuracy for the dataset.
        f1: Weighted F1 score for the dataset.
        all_labels: Ground truth labels.
        all_predictions: Predicted labels.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    start_time = time.time()

    with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                pbar.update(1)

    end_time = time.time()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Evaluation completed in {end_time - start_time:.2f}s")
    return epoch_loss, epoch_acc, f1, all_labels, all_predictions
