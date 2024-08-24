import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import pandas as pd
import os
from PIL import Image
from sklearn.metrics import f1_score
from tqdm import tqdm

print("Current working directory:", os.getcwd())

# 1. Load the CSV file
csv_path = 'train.csv'
print(f"Attempting to load CSV from: {os.path.abspath(csv_path)}")
df = pd.read_csv(csv_path)
print(f"Successfully loaded CSV with {len(df)} entries")

# 2. Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.image_dir, self.df.iloc[idx, 0] + '.jpg')
            image = Image.open(img_name).convert('RGB')
            label = self.df.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            return image, label
        except FileNotFoundError:
            print(f"Error: File not found: {img_name}")
            raise
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            raise

# 3. Enhanced Data Transformations with Augmentation
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Create Dataset and DataLoader
image_dir = 'Images'
print(f"Image directory: {os.path.abspath(image_dir)}")
full_dataset = ImageDataset(df, image_dir, transform=None)  # We'll apply transforms later
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Apply transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform

print(f"Dataset split - Train: {train_size}, Validation: {val_size}, Test: {test_size}")

# 5. Model (Vision Transformer)
def create_model(num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    # Replace the last layer of the classifier
    num_ftrs = model.heads[-1].in_features
    model.heads[-1] = nn.Linear(num_ftrs, num_classes)
    
    return model

# 6. Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_f1 = 0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val F1-Score: {val_f1:.4f}")
        
        scheduler.step(val_loss)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model_vit_h14.pth')

    return model

if __name__ == '__main__':
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=4)

    # Verify data loading
    print("Verifying data loading...")
    for images, labels in train_loader:
        print("Batch shape:", images.shape)
        print("Labels:", labels)
        break  # Just check the first batch

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    num_classes = 42  # Assuming 42 classes as in the original code
    model = create_model(num_classes)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Train model
    num_epochs = 100
    print(f"Starting training for {num_epochs} epochs...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # 8. Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load('best_model_vit.pth'))
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # 9. Calculate F1-Score
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    print('Test F1-Score:', test_f1)