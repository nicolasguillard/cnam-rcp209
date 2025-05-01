# %% [markdown]
# # Classification du jeu Imagenet-1K "samples" avec Alexnet
# 
# - Creation : *18/02/2025*
# 
# Constat de la performance de classification du modèle.
# Utilisation de la définition du modèle et des poids disponibles dans PyTorch.
# 
# - [ ] Essayer AlexNet_Weights.IMAGENET1K_V1.transforms, ensemble des transformations prèdéfinie dédié à AlexNet entrainé sur Imagenet-1K.
# - [ ] Evaluations de l'erreur de validation sur les jeux de données Imagenet-1K avec 1K images (imagenet-sample-images-master) et 50K images(imagenet_val_images) disponibles localement. 
# 

# %% [markdown]
# # Module

# %%
import os
from collections import Counter, defaultdict

import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from datasets import DATASET_1, DATASET_2, CustomImageDataset, get_label_data_from_filename
from utils.alexnet_for_deconv import alexnet_for_deconv
from utils.utils_images import display_image_tensor as display_image_tensor_
from imagenet_labels import imagenet1K_names_to_labels, imagenet1K_labels_to_names

# %%
# Spécifiquement pour un carnet de type Jupyter
def display_image_tensor(img_tensor, verbose=True):
    if display:
        display_image_tensor_(img_tensor, verbose=verbose, fn_display=display)

# %%
test_debug = False

# %% [markdown]
# # Device

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %% [markdown]
# # Chargement des données

# %% [markdown]
# Création du dataset sur les images

# %%
DATASET = DATASET_1

# %%
"""
from datasets import imagenet_mean, imagenet_std

geo_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
])

transforms = T.Compose([
    geo_transforms,
    T.Lambda(lambda t: t/255.), # because read_image -> [0..255]
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])
""";

transforms = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()

# Display the transforms
# Mais attention, contrairement à ce qui serait affiché, le resize est exécuté avant le crop
print("Transforms", transforms)

get_label_data = lambda f: get_label_data_from_filename(f, DATASET["path"])
dataset_path = DATASET["mounted_path"] if os.path.exists(DATASET["mounted_path"]) else DATASET["path"]
print("Dataset (name, path)", DATASET["name"], dataset_path)

dataset = CustomImageDataset(
    dataset_path,
    transform=transforms,
    extension="JPEG",
    dataset_mode=True,
    only_label_idx=False,
    get_label_data=get_label_data,
    )

# %%
dataset[1]

# %%
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %% [markdown]
# Test dataloader

# %%
if test_debug:
    i_element = 1
    batch = next(iter(dataloader))
    print("Dimension batch:", len(batch))
    
    print("Première partie du Batch :", len(batch[0]))
    print("Deuxième partie du Batch :", len(batch[1]))
    print("Troisième partie du Batch :", len(batch[2]))
    print("Quatrième partie du Batch :", len(batch[3]))

    print("Image :", batch[0][i_element].size())
    print("Idx etiquette :", batch[1][i_element].item())
    print("Code etiquette :", batch[2][i_element])
    print("Id file :", batch[3][i_element].item())

# %% [markdown]
# # Chargement du modèle

# %%
model_alexnet_deconv = alexnet_for_deconv(weights='IMAGENET1K_V1')
model_alexnet_deconv.eval()
model_alexnet_deconv.to(device)

# %%
if test_debug:
    i_image = 10
    batch_input = batch[0][i_image].unsqueeze(dim=0)
    print("Batch input :", batch_input.size())
    output = model_alexnet_deconv.forward(batch_input.to(device))

    print(output.size())
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted = probabilities.argmax(dim=1).to("cpu")
    print("predicted idx :", predicted)
    expected = batch[1][i_image]

    print("expected :", expected.tolist())

    #print(probabilities.argmax(dim=1).to("cpu") == batch[1][0])
    #print((probabilities.argmax(dim=1).to("cpu") == batch[1][0]).sum())

# %% [markdown]
# # Classification

# %% [markdown]
# Il faut espérer que l'indexation de la sortie du classifier correspond à celle des classes de l'ImageNet-1K récupérée dans la liste 1K.

# %%
distribution_expected = Counter()
distribution_computed = Counter()
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    output = model_alexnet_deconv.forward(batch[0].to(device))
    probabilities = torch.nn.functional.softmax(output, dim=0)
    predicted_labels = probabilities.argmax(dim=1).to("cpu")
    
    distribution_expected.update(batch[1].tolist())
    distribution_computed.update(predicted_labels.tolist())

# %%
def counter_to_tensor(counter, size=10):
    a = [0] * size
    for k, v in counter.items():
        a[k] = v
    return torch.tensor(a)

# Exemple d'utilisation
distribution_expected_t = counter_to_tensor(distribution_expected, len(imagenet1K_labels_to_names))
distribution_computed_t = counter_to_tensor(distribution_computed, len(imagenet1K_labels_to_names))
print("Equal count", (distribution_computed_t == distribution_expected_t).sum().item())
print("Equal ratio", (distribution_computed_t == distribution_expected_t).sum() / len(dataset))

print("Equal ratio by class", (distribution_computed_t - distribution_expected_t).abs() / torch.where(distribution_expected_t == 0, torch.tensor(1), distribution_expected_t))

#print(distribution_computed_t)
#print(distribution_expected_t)

# %%



