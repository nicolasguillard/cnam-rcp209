import matplotlib.pyplot as plt
import numpy as np

__version__ = "1.0.0"

def rgb_distributions(image_array, min_value=0, max_value=255, bins=256, density=False):
    """
    Process ditribution histograms of RGB channels in image (H, W, Channels)
    """
    R = image_array[:, :, 0]
    G = image_array[:, :, 1]
    B = image_array[:, :, 2]

    # Créer des histogrammes pour chaque canal
    bin_edges = np.linspace(min_value, max_value, bins)
    hist_R, _ = np.histogram(R, bins=bin_edges, density=density)
    hist_G, _ = np.histogram(G, bins=bin_edges, density=density)
    hist_B, _ = np.histogram(B, bins=bin_edges, density=density)

    return (bin_edges, hist_R, hist_G, hist_B)


def display_rgb_distributions_(bin_edges, hist_R, hist_G, hist_B, suf_title=""):
    """
    Display hisrograms
    """
    # Créer une figure pour les courbes
    plt.figure(figsize=(10, 6))

    # Afficher les courbes
    plt.step(bin_edges[:-1], hist_R, where='mid', color='red', label='Rouge')
    plt.step(bin_edges[:-1], hist_G, where='mid', color='green', label='Vert')
    plt.step(bin_edges[:-1], hist_B, where='mid', color='blue', label='Bleu')

    # Ajouter des légendes et des titres
    plt.title('Distribution des canaux RGB' + suf_title)
    plt.xlabel('Intensité')
    plt.ylabel('Densité')
    plt.legend()

    # Afficher le graphique
    plt.show()
    

def display_rgb_distributions(
        image_array, min_value=0, max_value=255, bins=256, suf_title="", density=False
    ):
    bin_edges, hist_R, hist_G, hist_B = rgb_distributions(
        image_array, min_value, max_value, bins, density
        )
    display_rgb_distributions_(bin_edges, hist_R, hist_G, hist_B, suf_title)