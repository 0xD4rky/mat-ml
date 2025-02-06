from model import *

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
f1_scores = []

import torch

best_f1_score = 0.0
best_model_path = "best_model.pth"

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
f1_scores = []

start_time = time.time()
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    val_loss, val_acc, f1, _, _ = evaluate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    f1_scores.append(f1)

    scheduler.step()

    if f1 > best_f1_score:
        best_f1_score = f1
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch + 1} with F1 Score: {f1:.4f}")

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1 Score: {f1:.4f}")

total_training_time = time.time() - start_time
print(f"Training completed in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
print("Loading the best model for testing...")
model.load_state_dict(torch.load(best_model_path))

test_loss, test_acc, test_f1, test_labels, test_predictions = evaluate(model, test_loader, criterion)

print("Test Accuracy: {:.4f}".format(test_acc))
print("Test F1 Score: {:.4f}".format(test_f1))
print("Classification Report:\n", classification_report(test_labels, test_predictions, target_names=class_names))


cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

plt.plot(range(1, num_epochs + 1), f1_scores, marker='o')
plt.title("F1 Score vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.show()