# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network was built for regression, with the dataset normalized and split into training and testing sets. The model was trained using RMSprop and MSE loss, and its performance was evaluated. A loss plot was generated, and predictions were made on new data.

## Neural Network Model

![image](https://github.com/user-attachments/assets/b1e9e406-ffc3-4972-97c0-2684219d58db)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Hemapriya K
### Register Number: 212223040066
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
   ai_brain.history = {'loss': []}
   history = {'loss': []}
   for epoch in range(epochs):
        y_pred = ai_brain(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history['loss'].append(loss.item())
        if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
    return history

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

history = train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

loss_df = pd.DataFrame(history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[11]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
    



```
## Dataset Information

![image](https://github.com/user-attachments/assets/2434f472-a030-46b7-81e8-7bcb23c91eef)


## OUTPUT

### Training Loss Vs Iteration Plot


![image](https://github.com/user-attachments/assets/2f1ac6eb-ba83-4935-994b-0f2f3a03f0b3)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/7c569656-580f-410c-8ead-fe5309637542)
![image](https://github.com/user-attachments/assets/0591a1e3-c090-4f35-98d0-b2bf63092b55)



## RESULT
Thus a neural network regression model for the dataset was developed successfully.
