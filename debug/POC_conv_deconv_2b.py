# %% [markdown]
# # Tests Conv Deconv (2b)
# 
# - creation: *17/02/2025*
# 
# Pour comparer avec (2a) qui sert de référence, en modifiant des paramètres.

# %% [markdown]
# ## Modules
display = None

# %%
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from torchvision.io import read_image, decode_image
from PIL import ImageDraw


# %%
print(torchvision.__version__)

# %%
from datasets import DATASET_0, DATASET_2
from utils import display_image_tensor as display_image_tensor_, display_pictures_grid, to_0_255, unnormalize
from utils import alexnetfordeconv, clean_feature_maps, deconvolution, get_output_sizes, get_receptive_field_in_pixel_space


# %%
# Spécifiquement pour un carnet de type Jupyter
def display_image_tensor(img_tensor, verbose=True):
    if display:
        display_image_tensor_(img_tensor, verbose=verbose, fn_display=display)

# %% [markdown]
# ## Devices

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
# ## Paramètres modifiés

# %%
ImageFile = "n04037443_racer.JPEG" #"n02391049_zebra.JPEG"
idx_layer = 9 # 2, 5, (7), (9), 14 // [1, 2, 3, 4, 5]
flip_kernels = False
use_bias = False
lcn_kernel_size = 5

# %% [markdown]
# ## Chargement d'une image particulière

# %%
DataPath = DATASET_2["path"] #DATASET_2["mounted_path"]
#ImageFile = "n02391049_zebra.JPEG"
filename = os.path.join(DataPath, ImageFile)
(left, top), (right, bottom) = (325, 150), (365, 190) # zone entourant l'oeil du zèbre

# %%
#img1_t = decode_image(filename)
img1_t = read_image(filename)
display_image_tensor(img1_t)

# %%
# Tensor space
crop_t = img1_t[:, top:bottom, left:right] # (C, H, W)
# PIL space
display_image_tensor(crop_t)

# %%
imagenet_mean = DATASET_0["means"] # Car utilisation des poids de imagenet-1K
imagenet_std = DATASET_0["stds"] # Car utilisation des poids de imagenet-1K

geometry_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
])

transforms = T.Compose([
    geometry_transforms,
    T.Lambda(lambda t: t/255.), # because read_image -> [0..255]
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# %% [markdown]
# Affichage de l'image après transformations géométriques appliquées.

# %%
img_tensor_tfd_geo = geometry_transforms(img1_t)
display_image_tensor(img_tensor_tfd_geo)

# %% [markdown]
# Affichage de l'image après toutes les transformations.

# %%
img_tensor_tfd = transforms(img1_t)
display_image_tensor(img_tensor_tfd)

# %% [markdown]
# Création d'un batch pour le test.

# %%
batch_input = img_tensor_tfd.unsqueeze(dim=0)

# %% [markdown]
# ## Chargement d'Alexnet et propagation (inférence) dans la partie "features" du modèle

# %% [markdown]
# Chargement.

# %%
model_alexnet = torchvision.models.alexnet(weights='IMAGENET1K_V1')
model_alexnet.eval()

# %% [markdown]
# Propagation

# %%
output_features = model_alexnet.features(batch_input).detach()
if display:
    display_pictures_grid(
        output_features.squeeze(dim=0).reshape(
            (output_features.size(1), 1, output_features.size(2), output_features.size(3))
            # .squeeze pour supprimer la partie "batch"
            # .reshape(...,1,...) pour insérer la dimension channel (à 1 car monochrome)
            ),
        per_rows=16,
        titles=range(output_features.size(1))
        )

# %% [markdown]
# ## Comparaison des applications du modèle fourni et de sa version dérivée pour la déconvolution

# %% [markdown]
# Création d'un modèle

# %%
model_alexnet_deconv = alexnetfordeconv(weights='IMAGENET1K_V1')
model_alexnet_deconv.eval()

# %% [markdown]
# ### Propagation sans indication d'une couche particulière (et donc la dernière par défaut), en utilisant la partie `features`.

# %% [markdown]
# Calcul et affichage de chaque carte obtenue dans cette couche.

# %%
output_features_deconv = model_alexnet_deconv.features(batch_input).detach()
print("Dimensions de la sortie de la déconvolution : ", output_features_deconv.size())
if display:
    display_pictures_grid(
        output_features_deconv.squeeze(dim=0).reshape(
            (output_features_deconv.size(1), 1, output_features_deconv.size(2), output_features_deconv.size(3))
            # .squeeze pour supprimer la partie "batch"
            # output_features_deconv.size(1) : nombre de canaux / cartes
            # .reshape(...,1,...) pour insérer la dimension channel (à 1 car monochrome)
            ),
        per_rows=16,
        titles=range(output_features_deconv.size(1))
        )

# %% [markdown]
# Vérifier que la sortie de la procédure développée pour la déconvolution est identique à celle du modèle issue de la procédure de génération de PyTorch.

# %%
assert torch.equal(output_features, output_features_deconv), f"Problème !"
print("OK !!!")

# %% [markdown]
# ### En traitant explicitement la dernière couche de la partie `features`

# %%
output_features_deconv, switches_indices = model_alexnet_deconv(batch_input, idx_layer=-1)
output_features_deconv = output_features_deconv.detach()
if display:
    display_pictures_grid(
        output_features_deconv.squeeze(dim=0).reshape(
            (output_features_deconv.size(1), 1, output_features_deconv.size(2), output_features_deconv.size(3))
            # .squeeze pour supprimer la partie "batch"
            # output_features_deconv.size(1) : nombre de canaux / cartes
            # .reshape(...,1,...) pour insérer la dimension channel (à 1 car monochrome)
            ),
        per_rows=16,
        titles=range(output_features_deconv.size(1))
        )

# %% [markdown]
# Vérifier que la sortie de la procédure développée pour la déconvolution est identique à celle du modèle issue de la procédure de génération de PyTorch.

# %%
assert torch.equal(output_features, output_features_deconv), f"Problème !"
print("OK !!!")

# %% [markdown]
# ## Déconvolution

# %% [markdown]
# ### Quelques tests préalables

# %% [markdown]
# Préalablement, détection du max de la sortie et de sa position dans les cartes de caractéristiques :

# %%
deconv_max = output_features_deconv.squeeze(dim=0).max()
print("max :", deconv_max)
print("position(s) dans le tenseur :", (output_features_deconv == deconv_max).nonzero())

# %% [markdown]
# Déconvolution de la sortie du dernier module de la partie `features` (c'est-à-dire sa dernière couche), càd la dernière couche.

# %%
print(output_features_deconv.size())
print(switches_indices[-1][1].size()) # Le dernier tuple (idx_module, indices), que les indices

# %% [markdown]
# Nettoyage des indices de pooling, en ne préservant que le max par batch

# %%
#cleaned_output, max_activation, coords = clean_feature_maps(
cleaned_output = clean_feature_maps(
    output_features_deconv,
    idx_map=False,
    pool_indices=switches_indices[-1][1],
    return_pos=True
)
print("position: ", cleaned_output[1])
cleaned_feature_maps = cleaned_output[0]
print("max :", cleaned_feature_maps.max())

# %% [markdown]
# ... ce qui correspond à la détection préalable.

# %% [markdown]
# Affichage des cartes nettoyées (en vérifiant la cohérence avec la position du max obtenue précédemment):

# %%
if display:
    display_pictures_grid(
        cleaned_feature_maps.squeeze(dim=0).reshape(
            (cleaned_feature_maps.size(1), 1, cleaned_feature_maps.size(2), cleaned_feature_maps.size(3))
            # .squeeze pour supprimer la partie "batch"
            # output_features_deconv.size(1) : nombre de canaux / cartes
            # .reshape(...,1,...) pour insérer la dimension channel (à 1 car monochrome)
            ),
        per_rows=16,
        titles=range(cleaned_feature_maps.size(1))
        )

# %% [markdown]
# A priori, en ne fournissant pas `idx_map`, la déconvolution repose sur l'ensemble des max de chaque canaux.

# %%
print(model_alexnet_deconv.features)
print(vars(model_alexnet_deconv))
print(hasattr(model_alexnet_deconv, "features"))

# %% [markdown]
# ### Déconvolution

# %%
#output, max_coords = deconvolution(
output = deconvolution(
    model_alexnet_deconv,
    batch_input,
    idx_layer=idx_layer,
    flip_kernels=flip_kernels,
    use_bias=use_bias,
    clean_feature_map=True,
    idx_map=False,
    return_pos=True,
    verbose=True
    )

# %%
output_max_pos = output[1]
print(output_max_pos)

# %%
batch_deconvolutions = output[0]
display_image_tensor(batch_deconvolutions[0]) # First of the batch

# %% [markdown]
# ### Calcul du champ réceptif

# %%
output_sizes = get_output_sizes(model_alexnet_deconv.features, input_size=batch_input.size())
output_sizes

# %%
pixel_space_size = batch_input.size(-2), batch_input.size(-1)
max_pos = output_max_pos[0] # max of this batch
pos = max_pos[2:].tolist()
print(pos)
receptive_field = get_receptive_field_in_pixel_space(
    pos=pos,
    idx_layer=idx_layer,
    cnn_modules=model_alexnet_deconv.features,
    output_sizes=output_sizes,
    pixel_space_size=pixel_space_size
)
print(receptive_field)
((top, left), (bottom, right)) = receptive_field

# %%
img_input = T.functional.to_pil_image(batch_input[0])
img_draw = ImageDraw.Draw(img_input)
img_draw.rectangle([(left, top), (right, bottom)], outline="white")
if display:
    display(img_input)

# %%
deconvolution = batch_deconvolutions[0].detach()

# %% [markdown]
# Affichage de la déconvolution et du cadre du champ réceptif

# %%
img_deconv = T.functional.to_pil_image(deconvolution.clone())
img_draw = ImageDraw.Draw(img_deconv)
img_draw.rectangle([(left, top), (right, bottom)], outline="white")
if display:
    display(img_deconv)

# %%
print(deconvolution.min(), deconvolution.max(), deconvolution.mean())

# %% [markdown]
# ### Restitution de la déconvolution dans le champ réception et appréciation du résultat pour la visualisation de caractéristique.

# %% [markdown]
# Affichage que la partie de la déconvolution correspondant au champ réceptif

# %%
display_image_tensor(deconvolution[:, top:bottom, left:right])

# %% [markdown]
# Modification de la déconvolution pour essayer de rendre l'affichage de caractéristique plus expressif :
# - application d'une ReLU

# %%
display_image_tensor(nn.functional.relu(deconvolution[:, top:bottom, left:right]))

# %% [markdown]
# - application d'une ReLU et redistribution sur $[0; 255]$ :

# %%
display_image_tensor(to_0_255(nn.functional.relu(deconvolution[:, top:bottom, left:right])))

# %% [markdown]
# - Simplement redistribution sur $[0; 255]$ :

# %%
display_image_tensor(to_0_255(deconvolution[:, top:bottom, left:right]))

# %% [markdown]
# - application d'une "dénormalisation" inverse à la normalisation appliquée aux images en entrée

# %%
display_image_tensor(
    unnormalize(deconvolution[:, top:bottom, left:right], mean=imagenet_mean, std=imagenet_std)
)

# %% [markdown]
# - application d'une "dénormalisation" inverse à la normalisation appliquée aux images en entrée, après une ReLU :

# %%
display_image_tensor(
    unnormalize(
        nn.functional.relu(deconvolution[:, top:bottom, left:right]),
        mean=imagenet_mean, std=imagenet_std
    )
)

# %% [markdown]
# - Utilisation d'une LocalContrastNormalization

# %%
from utils.local_contrast_normalization import LocalContrastNormalization
local_contrast_normalization = LocalContrastNormalization(lcn_kernel_size)
display_image_tensor(local_contrast_normalization(deconvolution[:, top:bottom, left:right]))

# %% [markdown]
# - Utilisation d'une LocalContrastNormalization après une ReLU

# %%
display_image_tensor(local_contrast_normalization(
    nn.functional.relu(deconvolution[:, top:bottom, left:right])
))

# %% [markdown]
# - LocalContrastNormalization puis "dénormalisation"

# %%
display_image_tensor(
    unnormalize(
        local_contrast_normalization(deconvolution[:, top:bottom, left:right]),
        mean=imagenet_mean, std=imagenet_std
    )
)


