import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Get information about house prices in California
data = fetch_california_housing()
# Split our data into two parts:
# X = information about houses (like size, location, etc.)
# y = the actual house prices we want to predict
X, y = data.data, data.target 

# Make all our numbers similar in size
# Think of it like converting different measurements (inches, feet, miles) 
# all into the same unit so they're easier to compare
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split our data into two groups:
# - One group to teach our computer (training data, 80%)
# - One group to test how well it learned (testing data, 20%)
# We use 42 as a "seed" number so we get the same split every time we run the program
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert our data into a special format that PyTorch can understand
# It's like translating our numbers into a language the computer prefers
X_train = torch.tensor(X_train, dtype=torch.float32)
# Reshape our data so it's in the right format
# Like organizing items neatly in a container
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class HousePriceNN(nn.Module):
    def __init__(self):
        super(HousePriceNN, self).__init__()
        # Think of this like a chain of workers passing information:
        # 8 workers pass info to 128 workers, then to 64, then to 32, and finally to 1
        self.fc1 = nn.Linear(X.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        # ReLU is like an on/off switch - if the number is negative, make it zero
        # if it's positive, keep it as is
        self.relu = nn.ReLU()
        # Dropout is like randomly taking breaks
        # It helps our computer not memorize the answers but actually learn
        # Like covering up some notes while studying to test your real understanding
        self.dropout = nn.Dropout(0.2)  # 20% of the time, take a break

    def forward(self, x):
        # Apply layer operations with activation, dropout between each layer
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)  # No activation on output layer for regression
        return x

# Training setup
model = HousePriceNN()
# MSE Loss is like measuring how far off our guesses are
# If we guess a house costs $300k but it's really $350k, that's an error we want to minimize
criterion = nn.MSELoss()
# Adam is like a smart teacher that helps adjust how quickly our computer learns
# lr=0.001 is like taking small careful steps while learning
optimizer = optim.Adam(model.parameters(), lr=0.001)
# This is like a teacher who makes the lessons easier or harder
# depending on how well the student (our model) is doing
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # If student is struggling, make the lessons easier
    factor=0.5,      # Make the lessons half as difficult
    patience=20,     # Wait 20 tries before making it easier
    verbose=True     # Tell us when we're making changes
)

# Organize our data into small groups (batches) of 32
# It's like studying flashcards in small groups instead of all at once
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True    # Mix up the order, like shuffling flashcards
)

# Training settings
epochs = 1000  # We'll practice 1000 times
best_loss = float('inf')  # Keep track of our best score (smaller is better)
patience = 50    # How many times we'll try to improve before giving up
patience_counter = 0  # Counting how many times we haven't improved

# Training loop
for epoch in range(epochs):
    model.train()  # Enable dropout and batch normalization
    total_loss = 0 # Total loss of the model.
    
    # Batch training
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()        # Reset gradients. Gradient is the slope of the loss function.
        outputs = model(batch_X)     # Forward pass. It is the process of passing the input data through the model to get the output.
        loss = criterion(outputs, batch_y)  # Calculate loss. Loss is the difference between the predicted output and the actual output.
        loss.backward()             # Backward pass. It is the process of calculating the gradient of the loss function. 
        optimizer.step()            # Update weights. It is the process of updating the weights of the model.
        total_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader) # Average loss is used to measure the performance of the model.
    # Update learning rate if needed. The step uses the average loss to update the learning rate by a factor of 0.5.
    # As an example, if the learning rate is 0.001, it will be reduced to 0.0005.
    scheduler.step(avg_loss)  
    
    # Early stopping logic
    # If the loss is less than the best loss, then the best loss is updated and the patience counter is reset.
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    # If the patience counter is greater than the patience, then the model is stopped, because it is not performing well.
    # As an example, if the patience is 50, and the model is not performing well, it will stop at epoch 50.
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
    
    # Print progress every 50 epochs
    if epoch % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Model evaluation is the process of evaluating the performance of the model. It's used to check if the model is performing well.
# You would switch to training mode when you want to train the model and switch to evaluation mode when you want to evaluate the model.
model.eval()  # Disable dropout for evaluation
with torch.no_grad():  # Disable gradient calculation because we are not training the model.
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Save what our computer learned to a file
# Like saving your homework to work on it later
torch.save(model.state_dict(), 'model.pth')

# This is how you would load the saved work later
# (It's commented out because we don't need it right now)
# model = HousePriceNN()
# model.load_state_dict(torch.load('model.pth'))


