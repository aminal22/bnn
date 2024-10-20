import torch
from torchvision import transforms
from PIL import Image
import util  # Assuming util contains your BinOp class
from models import LeNet_5

# Define if you are using CUDA or not
cuda_available = torch.cuda.is_available()

# Load the trained model
model = LeNet_5()  # or whatever architecture you used
model.load_state_dict(torch.load('models/LeNet_5.best.pth.tar')['state_dict'])
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to the size expected by MNIST
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Same normalization as training
])

# Load and preprocess your image
img_path = 'huit.jpg'  # Replace with the path to your image
image = Image.open(img_path).convert('L')  # Convert to grayscale
image = transform(image).unsqueeze(0)  # Add a batch dimension

# If using CUDA, move the image to the GPU
if cuda_available:
    image = image.cuda()
    model.cuda()  # Move the model to GPU as well

# Make predictions
with torch.no_grad():
    output = model(image)
    pred = output.data.max(1, keepdim=True)[1]  # Get the predicted class

# Print the prediction
print(f'Predicted class: {pred.item()}')
