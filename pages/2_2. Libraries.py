import streamlit as st

tab1, tab2, tab3 = st.tabs(["Pandas", "NumPy", "PyTorch"])

with tab1:
    st.header('Pandas', divider='rainbow')
    st.image("assets/PandasLogo.png", use_container_width=True)
    st.markdown(f'''

## 1. **Installation and Setup**

### Installation
```bash
pip install pandas
```

### Importing Pandas
```python
import pandas as pd

## 2. **Core Data Structures**

### Series
A one-dimensional labeled array capable of holding data of any type.
```python
s = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
print(s)
```

### DataFrame
A two-dimensional, size-mutable, and potentially heterogeneous data structure.
```python
data = 
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]

df = pd.DataFrame(data)
print(df)
```

---

## 3. **Reading and Writing Data**

### Reading Data
```python
# CSV
df = pd.read_csv("data.csv")

# Excel
df = pd.read_excel("data.xlsx")

# JSON
df = pd.read_json("data.json")
```

### Writing Data
```python
# CSV
df.to_csv("output.csv", index=False)

# Excel
df.to_excel("output.xlsx", index=False)

# JSON
df.to_json("output.json")
```

---

## 4. **Viewing and Inspecting Data**

### Displaying Data
```python
df.head()       # First 5 rows
df.tail()       # Last 5 rows
df.info()       # Summary of the DataFrame
df.describe()   # Descriptive statistics
```

### Shape and Columns
```python
df.shape         # (rows, columns)
df.columns       # List of column names
df.dtypes        # Data types of columns
```

---

## 5. **Selecting and Filtering Data**

### Selecting Columns
```python
df["ColumnName"]      # Single column
df[["Col1", "Col2"]]  # Multiple columns
```

### Selecting Rows
```python
df.iloc[0]            # By position
df.loc[0]             # By index
```

### Filtering Rows
```python
df[df["Age"] > 30]    # Rows where Age > 30
df[(df["Age"] > 30) & (df["City"] == "New York")]  # Multiple conditions
```

---

## 6. **Modifying Data**

### Adding/Updating Columns
```python
df["NewCol"] = df["Age"] * 2
```

### Dropping Columns or Rows
```python
df.drop("ColumnName", axis=1, inplace=True)  # Drop column
df.drop(0, axis=0, inplace=True)            # Drop row
```

### Renaming Columns
```python
df.rename(columns="OldName": "NewName", inplace=True)
```

---

## 7. **Missing Data**

### Identifying Missing Values
```python
df.isnull()      # Boolean DataFrame of missing values
df.isnull().sum()  # Count missing values per column
```

### Filling Missing Values
```python
df.fillna(0, inplace=True)      # Fill with a specific value
df.fillna(df.mean(), inplace=True)  # Fill with column mean
```

### Dropping Missing Values
```python
df.dropna(inplace=True)  # Drop rows with missing values
```

---

## 8. **Data Operations**

### Sorting
```python
df.sort_values("ColumnName", ascending=True, inplace=True)
```

### Grouping and Aggregating
```python
df.groupby("Category").sum()  # Group by 'Category' and sum values
```

### Applying Functions
```python
df["NewCol"] = df["Col"].apply(lambda x: x * 2)
```

---

## 9. **Merging, Joining, and Concatenating**

### Concatenating
```python
result = pd.concat([df1, df2], axis=0)  # Vertical concatenation
```

### Merging
```python
merged = pd.merge(df1, df2, on="KeyColumn", how="inner")
```

### Joining
```python
joined = df1.join(df2, how="outer")
```

---

## 10. **Indexing**

### Setting/Resetting Index
```python
df.set_index("ColumnName", inplace=True)
df.reset_index(inplace=True)
```

### Multi-Indexing
```python
df.set_index(["Col1", "Col2"], inplace=True)
```

---

## 11. **Working with Time Series**

### Converting to Datetime
```python
df["Date"] = pd.to_datetime(df["Date"])
```

### Resampling
```python
df.resample("M").mean()  # Resample to monthly frequency
```

### Date Operations
```python
df["Year"] = df["Date"].dt.year
```

---

## 12. **Essential Functions**

| Function                  | Description                                  |
|---------------------------|----------------------------------------------|
| `pd.concat()`             | Concatenate DataFrames                      |
| `pd.merge()`              | Merge DataFrames                            |
| `df.groupby()`            | Group data                                  |
| `df.sort_values()`        | Sort rows                                   |
| `df.pivot()`              | Pivot data                                  |
| `df.pivot_table()`        | Create pivot table                          |
| `pd.to_datetime()`        | Convert to datetime                         |
| `df.corr()`               | Correlation matrix                          |
| `df.duplicated()`         | Find duplicates                             |
| `df.drop_duplicates()`    | Remove duplicates                           |

---

## 13. **Plotting with Pandas**

### Built-in Plotting
```python
import matplotlib.pyplot as plt

df.plot(kind="line")
df["ColumnName"].plot(kind="hist")
plt.show()
```

---

## 14. **Performance Tips**

- Use `df.sample()` for sampling rows.
- Use `.astype()` to convert data types for optimization.
- Utilize `@pd.api.extensions.register_dataframe_accessor` for custom DataFrame methods.

---

## 15. **Common Commands**

| Command                     | Description                              |
|-----------------------------|------------------------------------------|
| `df.head()`                 | First 5 rows                            |
| `df.tail()`                 | Last 5 rows                             |
| `df.info()`                 | DataFrame summary                       |
| `df.describe()`             | Summary statistics                      |
| `df.isnull()`               | Identify missing values                 |
| `df.dropna()`               | Drop missing values                     |
| `df.fillna()`               | Fill missing values                     |
| `df.sort_values()`          | Sort rows by column                     |
                ''')

with tab2:
    st.header('NumPy', divider='rainbow')
    st.image("assets/NumPyLogo.png", use_container_width=True)
    st.markdown(f'''


## 1. **Installation and Setup**

### Installation
```bash
pip install numpy
```

### Importing NumPy
```python
import numpy as np
```

---

## 2. **Creating Arrays**

### 1D Array
```python
arr = np.array([1, 2, 3, 4])
```

### 2D Array
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
```

### 3D Array
```python
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

---

## 3. **Array Initialization**

### Predefined Arrays
```python
np.zeros((3, 3))        # 3x3 array of zeros
np.ones((2, 2))         # 2x2 array of ones
np.full((2, 3), 7)      # 2x3 array filled with 7
np.eye(3)               # Identity matrix of size 3
```

### Ranges
```python
np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)    # [0. 0.25 0.5 0.75 1.]
```

### Random Arrays
```python
np.random.rand(3, 2)    # Random values in [0, 1]
np.random.randn(3, 2)   # Standard normal distribution
np.random.randint(0, 10, (2, 2))  # Random integers
```

---

## 4. **Array Properties**

### Shape and Dimensions
```python
arr.shape         # Shape of the array
arr.size          # Total number of elements
arr.ndim          # Number of dimensions
```

### Data Type
```python
arr.dtype         # Data type of elements
arr.astype('float32')  # Convert data type
```

---

## 5. **Indexing and Slicing**

### Indexing
```python
arr[1]            # Access second element
arr[1, 2]         # Access element at row 1, col 2
```

### Slicing
```python
arr[0:2]          # First two elements
arr[:, 1]         # All rows, second column
arr[1:, :2]       # Rows 1 to end, columns 0 to 1
```

### Boolean Indexing
```python
arr[arr > 5]      # Elements greater than 5
```

---

## 6. **Array Operations**

### Arithmetic
```python
arr + 2           # Add 2 to all elements
arr * 3           # Multiply all elements by 3
arr1 + arr2       # Element-wise addition
arr1 * arr2       # Element-wise multiplication
```

### Statistical Operations
```python
arr.sum()         # Sum of all elements
arr.mean()        # Mean of elements
arr.std()         # Standard deviation
arr.min()         # Minimum value
arr.max(axis=0)   # Max along columns
```

### Aggregations
```python
np.cumsum(arr)    # Cumulative sum
np.prod(arr)      # Product of all elements
np.unique(arr)    # Unique elements
```

---

## 7. **Reshaping Arrays**

### Changing Shape
```python
arr.reshape(3, 2)      # Reshape to 3x2
arr.flatten()          # Flatten array
```

### Stacking
```python
np.vstack([arr1, arr2])  # Vertical stacking
np.hstack([arr1, arr2])  # Horizontal stacking
```

---

## 8. **Broadcasting**

NumPy automatically expands arrays with smaller dimensions to match larger ones.
```python
arr + np.array([1, 2, 3])  # Add a 1D array to a 2D array
```

---

## 9. **Linear Algebra**

### Matrix Operations
```python
np.dot(arr1, arr2)      # Matrix multiplication
np.transpose(arr)       # Transpose of a matrix
np.linalg.inv(arr)      # Inverse of a matrix
```

### Eigenvalues and Eigenvectors
```python
np.linalg.eig(arr)
```

### Solving Linear Equations
```python
np.linalg.solve(A, b)
```

---

## 10. **Random Module**

### Seed for Reproducibility
```python
np.random.seed(42)
```

### Random Sampling
```python
np.random.choice(arr, size=3, replace=False)  # Random sample without replacement
```

---

## 11. **Special Functions**

### Mathematical Operations
```python
np.sqrt(arr)          # Square root
np.log(arr)           # Natural log
np.exp(arr)           # Exponent
np.sin(arr)           # Sine
np.cos(arr)           # Cosine
np.round(arr, 2)      # Round to 2 decimals
```

---

## 12. **Handling Missing Data**

### Identifying Missing Values
```python
np.isnan(arr)        # Check for NaN values
```

### Replacing Missing Values
```python
arr[np.isnan(arr)] = 0  # Replace NaN with 0
```

---

## 13. **File Input/Output**

### Saving and Loading Arrays
```python
np.save("array.npy", arr)       # Save as .npy file
arr = np.load("array.npy")      # Load .npy file
```

### Text Files
```python
np.savetxt("array.txt", arr)    # Save as text file
arr = np.loadtxt("array.txt")   # Load text file
```

---

## 14. **Performance Tips**

- Use `np.vectorize()` for element-wise operations.
- Use broadcasting instead of Python loops for speed.
- Use `np.memmap` for large datasets to avoid memory overload.

---

## 15. **Common Commands**

| Command                     | Description                                  |
|-----------------------------|----------------------------------------------|
| `np.zeros()`                | Create an array of zeros                    |
| `np.ones()`                 | Create an array of ones                     |
| `np.arange()`               | Create an array with evenly spaced values   |
| `np.linspace()`             | Create an array with specific number of points between two values |
| `np.mean()`                 | Calculate the mean                          |
| `np.sum()`                  | Sum all elements                            |
| `np.cumsum()`               | Cumulative sum                             |
| `np.dot()`                  | Matrix multiplication                       |

---

## 16. **Helpful Tips**

- NumPy integrates seamlessly with libraries like Pandas, Matplotlib, and SciPy.
- Use `np.array()` to convert lists to arrays for efficiency.
- Optimize loops with array operations wherever possible.

---

## 17. **Example Workflow**

```python
import numpy as np

# Step 1: Create an array
arr = np.random.randint(1, 10, size=(3, 3))

# Step 2: Perform operations
arr_sum = arr.sum(axis=0)
arr_squared = np.square(arr)

# Step 3: Save results
np.save("results.npy", arr_squared)

# Step 4: Load and print results
loaded_arr = np.load("results.npy")
print(loaded_arr)
```
                ''')

with tab3:
    st.header('PyTorch', divider='rainbow')
    st.image("assets/PyTorchLogo.png", use_container_width=True)
    st.markdown(f'''

## 1. **Installation and Setup**

### Installation
```bash
pip install torch torchvision torchaudio
```

### Importing PyTorch
```python
import torch
```

---

## 2. **Tensors**

### Creating Tensors
```python
# From a list
x = torch.tensor([[1, 2], [3, 4]])

# With specific data types
x = torch.tensor([1.0, 2.0], dtype=torch.float32)

# Predefined tensors
torch.zeros((2, 2))      # 2x2 tensor of zeros
torch.ones((3, 3))       # 3x3 tensor of ones
torch.rand((4, 4))       # Random values between 0 and 1
torch.eye(3)             # Identity matrix
```

### Converting Between NumPy and Tensors
```python
import numpy as np

# NumPy to PyTorch
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# PyTorch to NumPy
np_array_back = tensor.numpy()
```

---

## 3. **Tensor Operations**

### Basic Arithmetic
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

a + b           # Element-wise addition
a * b           # Element-wise multiplication
a @ b.T         # Matrix multiplication
```

### Reduction Operations
```python
a.sum()         # Sum of all elements
a.mean()        # Mean of all elements
a.min()         # Minimum value
a.max()         # Maximum value
a.argmax()      # Index of maximum value
```

### Reshaping
```python
a.view(2, 3)    # Reshape to 2x3
a.reshape(-1)   # Flatten tensor
a.transpose(0, 1)  # Swap axes
```

---

## 4. **Indexing and Slicing**

### Basic Indexing
```python
x = torch.tensor([[1, 2], [3, 4]])
x[0]             # First row
x[:, 1]          # Second column
x[1, 1]          # Element at row 1, col 1
```

### Boolean Indexing
```python
x[x > 2]         # Elements greater than 2
```

---

## 5. **Device Management**

### Checking for GPU
```python
torch.cuda.is_available()
```

### Moving Tensors Between Devices
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3])
x = x.to(device)      # Move tensor to GPU
x = x.cpu()           # Move tensor back to CPU
```

---

## 6. **Autograd (Automatic Differentiation)**

### Enabling Gradient Computation
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
```

### Computing Gradients
```python
y = x * 2
z = y.mean()
z.backward()         # Compute gradients
print(x.grad)        # Access gradients
```

### Disabling Gradient Computation
```python
with torch.no_grad():
    y = x * 2
```

---

## 7. **Neural Networks**

### Defining a Model
```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
```

### Model Summary
```python
print(model)
```

---

## 8. **Loss Functions**
```python
criterion = nn.MSELoss()       # Mean Squared Error
loss = criterion(predicted, target)
```

---

## 9. **Optimizers**
```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.step()              # Perform a single optimization step
optimizer.zero_grad()         # Reset gradients
```

---

## 10. **Data Loading**

### Dataset and DataLoader
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(torch.rand(100, 10), torch.rand(100, 1))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    inputs, labels = batch
```

---

## 11. **Common Utilities**

### Saving and Loading Models
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Tensor Operations
```python
torch.cat([a, b], dim=0)   # Concatenate tensors along dimension
torch.stack([a, b])        # Stack tensors along a new dimension
```

---

## 12. **Example Workflow**

### Simple Linear Regression
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Generate data
x = torch.rand(100, 1)
y = 2 * x + 1 + torch.randn(100, 1) * 0.1

# Step 2: Define model
model = nn.Linear(1, 1)

# Step 3: Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Step 4: Training loop
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(x)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch epoch+1, Loss: loss.item()")
```
                ''')