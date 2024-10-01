"""
Makes Prediction on a custom image and then plots is.
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List
import model_builder


data_transforms = transforms.Compose([
  transforms.Resize((64, 64))
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        device: torch.device,
                        class_names: List[str] = None,
                        transform=None):
  """Makes a prediction on a target image and plots image with its prediction."""

  # 1. Load in image and convert the tensor values to float32
  target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

  # 2. Divide the image pixel values by 255 to get them between [0, 1]
  target_image = target_image / 255.

  # 3. Transform if necessary
  if transform:
    target_image = transform(target_image)

  # 4. Make sure the model is on target device
  model.to(device)

  # 5. Turn on model evaluation mode and inference mode
  model.eval()
  with torch.inference_mode():
    # Add an extra dimension to the image
    target_image = target_image.unsqueeze(dim=0)

    # Make a prediction on image with an extra dimension and send it to target device
    target_image_pred = model(target_image.to(device))

  # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
  target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

  # 7. Convert prediction probabilities -> prediction labels
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

  # 8. Plot the image alongside the prediction and prediction probability
  plt.imshow(target_image.squeeze().permute(1, 2, 0)) # Make sure it's the right size for matplotlib
  if class_names:
    title=f'Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}'
  else:
    title=f'Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}'
  plt.title(title)
  plt.axis(False)
  plt.savefig('./pred_plot.png')

model_dir = Path('models/')
model_path = model_dir / 'tinyvgg_model.pth'

loaded_model = model_builder.TinyVGG(3, 10, 3)
loaded_model.load_state_dict(torch.load(f=model_path, weights_only=True))

pred_and_plot_image(model=loaded_model,
                    image_path='./data/pizza_image.jpg',
                    class_names=['pizza', 'steak', 'sushi'],
                    transform=data_transforms,
                    device=device)