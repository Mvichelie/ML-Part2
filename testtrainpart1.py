import json
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize
from wikiart import WikiArtDataset, WikiArtModel
import torcheval.metrics as metrics
import matplotlib.pyplot as plt 


with open("config.json", "r") as f:
    config = json.load(f)


transform = Compose([
    Resize((224, 224)),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = WikiArtDataset(config["testingdir"], transform=transform, device=config["device"])
dataloader = DataLoader(dataset, batch_size=1) 

#trained model
model = WikiArtModel(num_classes=len(dataset.classes), bonusA=True).to(config["device"])
model.load_state_dict(torch.load(config["modelsavedfile1"])) 
model.eval()  

accuracy_metric = metrics.MulticlassAccuracy()
precision_metric = metrics.MulticlassPrecision(num_classes=len(dataset.classes), average="macro")
recall_metric = metrics.MulticlassRecall(num_classes=len(dataset.classes), average="macro")
f1_metric = metrics.MulticlassF1Score(num_classes=len(dataset.classes), average="macro")
confusion_matrix = metrics.MulticlassConfusionMatrix(num_classes=len(dataset.classes))

#test
def test_classification():
    """
    Evaluate the classification model on the testing dataset.
    """
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(config["device"])
            labels = labels.to(config["device"])

            #forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            #updated metrics
            accuracy_metric.update(predictions, labels)
            precision_metric.update(predictions, labels)
            recall_metric.update(predictions, labels)
            f1_metric.update(predictions, labels)
            confusion_matrix.update(predictions, labels)

    #evaluation results
    print(f"Test Accuracy: {accuracy_metric.compute()}")
    print(f"Precision : {precision_metric.compute()}")
    print(f"Recall : {recall_metric.compute()}")
    print(f"F1 Score : {f1_metric.compute()}")

    #visualization of confusion matrix
    confusion = confusion_matrix.compute().cpu().numpy()
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")

test_classification()
