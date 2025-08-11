import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict

from models.vae import VAE
from models.vq_codebook import GeodesicVectorQuantizer
from utils.dataloader import get_mnist_single_batch
from utils.device import device


def create_and_test_codebook():
    """
    Funzione principale per creare e testare il codebook geodesico.
    """
    print("="*80)
    print("CREAZIONE E TEST DEL CODEBOOK GEODESICO")
    print("="*80)

    # 1. Configurazione
    clustering_path = "output/clustering_knn_graphs/clustering_32_18_128.npz"
    vae_model_path = "output/vae/vae_model_32d.pth"

    print(f"File cluster: {clustering_path}")
    print(f"Modello VAE: {vae_model_path}")

    # Verifica che i file esistano
    if not Path(clustering_path).exists():
        print(f"File cluster non trovato: {clustering_path}")
        return False

    if not Path(vae_model_path).exists():
        print(f"Modello VAE non trovato: {vae_model_path}")
        return False

    # 2. Creazione del Vector Quantizer
    print(f"\n" + "="*60)
    print("CREAZIONE DEL VECTOR QUANTIZER")
    print("="*60)

    # Carica i dati del clustering per determinare le dimensioni
    clustering_data = np.load(clustering_path, allow_pickle=True)
    config = clustering_data['config'].item()

    print(f"Configurazione del clustering:")
    print(f"  • Dimensione latente: {config['latent_dim']}")
    print(f"  • k-neighbors: {config['k_neighbors']}")
    print(f"  • Numero cluster: {config['n_clusters']}")

    # Inizializza il Vector Quantizer
    quantizer = GeodesicVectorQuantizer(
        codebook_size=config['n_clusters'],
        embedding_dim=config['latent_dim'],
        commitment_cost=0.25
    )

    # Carica il codebook dal clustering
    quantizer.load_from_clustering_results(clustering_path)

    # Calcola le distanze geodesiche tra i vettori del codebook
    quantizer.compute_geodesic_distances_between_codes(
        k_neighbors=config["k_neighbors"]
    )

    print("Codebook creato con successo!")

    # 3. Test della quantizzazione su nuovi dati
    print(f"\n" + "="*60)
    print("TEST DELLA QUANTIZZAZIONE")
    print("="*60)

    # Carica il modello VAE
    vae_model = VAE(input_dim=784, hidden_dim=400, latent_dim=config['latent_dim'])
    vae_checkpoint = torch.load(vae_model_path, map_location=device)
    vae_model.load_state_dict(vae_checkpoint)
    vae_model.to(device)
    vae_model.eval()

    # Carica nuovi dati di test
    test_data, test_labels = get_mnist_single_batch(max_samples=1000, split='test')
    test_data = test_data.to(device)

    print(f"Dati di test caricati: {test_data.shape}")

    # Estrai rappresentazioni latenti
    with torch.no_grad():
        mu, logvar = vae_model.encode(test_data.view(-1, 784))
        latent_vectors = mu.cpu()

    # Test quantizzazione euclidea
    print("\nQuantizzazione Euclidea...")
    start_time = time.time()
    quantized_euc, loss_euc, indices_euc = quantizer.quantize(latent_vectors, use_geodesic=False)
    euclidean_time = time.time() - start_time

    # Test quantizzazione geodesica
    print("Quantizzazione Geodesica...")
    start_time = time.time()
    quantized_geo, loss_geo, indices_geo = quantizer.quantize(latent_vectors, use_geodesic=True)
    geodesic_time = time.time() - start_time

    # 4. Analisi dei risultati
    print(f"\n" + "="*60)
    print("ANALISI DEI RISULTATI")
    print("="*60)

    # Statistiche temporali
    print(f"PRESTAZIONI:")
    print(f"  • Tempo quantizzazione euclidea: {euclidean_time:.3f}s")
    print(f"  • Tempo quantizzazione geodesica: {geodesic_time:.3f}s")
    print(f"  • Overhead geodesico: {(geodesic_time/euclidean_time - 1)*100:.1f}%")

    # Statistiche di loss
    print(f"\nQUALITÀ QUANTIZZAZIONE:")
    print(f"  • Loss euclidea: {loss_euc.item():.4f}")
    print(f"  • Loss geodesica: {loss_geo.item():.4f}")
    print(f"  • Miglioramento loss: {((loss_euc - loss_geo)/loss_euc*100).item():.1f}%")

    # Utilizzo del codebook
    usage_euc = quantizer.get_codebook_usage(indices_euc)
    usage_geo = quantizer.get_codebook_usage(indices_geo)

    print(f"\nUTILIZZO CODEBOOK:")
    print(f"  • Euclidea - Codici utilizzati: {usage_euc['used_codes']}/{usage_euc['total_codes']} ({usage_euc['usage_rate']:.1%})")
    print(f"  • Geodesica - Codici utilizzati: {usage_geo['used_codes']}/{usage_geo['total_codes']} ({usage_geo['usage_rate']:.1%})")

    # Differenze nell'assegnazione
    different_assignments = (indices_euc != indices_geo).sum().item()
    agreement_percentage = (1 - different_assignments / len(indices_euc)) * 100

    print(f"\nCONFRONTO ASSEGNAZIONI:")
    print(f"  • Assegnazioni identiche: {len(indices_euc) - different_assignments}/{len(indices_euc)} ({agreement_percentage:.1f}%)")
    print(f"  • Assegnazioni diverse: {different_assignments}/{len(indices_euc)} ({100 - agreement_percentage:.1f}%)")

    # 5. Salvataggio
    print(f"\n" + "="*60)
    print("SALVATAGGIO")
    print("="*60)

    # Salva il codebook
    codebook_path = "output/geodesic_codebook.pth"
    quantizer.save_codebook(codebook_path)

    print(f"Codebook salvato: {codebook_path}")

    # 6. Visualizzazioni
    print(f"\n" + "="*60)
    print("GENERAZIONE VISUALIZZAZIONI")
    print("="*60)

    # Visualizza il codebook
    quantizer.visualize_codebook("output/codebook_visualization.png")

    # Crea visualizzazione della distribuzione di utilizzo
    create_usage_visualization(usage_euc, usage_geo, "output/codebook_usage_comparison.png")

    # 7. Riepilogo finale
    print(f"\n" + "="*80)
    print("RIEPILOGO FINALE")
    print("="*80)

    print(f"CODEBOOK GEODESICO CREATO CON SUCCESSO!")
    print(f"Statistiche principali:")
    print(f"   • Dimensione codebook: {quantizer.codebook_size} vettori")
    print(f"   • Dimensione embedding: {quantizer.embedding_dim}D")
    print(f"   • Miglioramento loss quantizzazione: {((loss_euc - loss_geo)/loss_euc*100).item():.1f}%")
    print(f"   • Accordo euclidea-geodesica: {agreement_percentage:.1f}%")
    print(f"   • Utilizzo codebook (geodesica): {usage_geo['usage_rate']:.1%}")

    print(f"\nFile creati:")
    print(f"   • {codebook_path}")
    print(f"   • output/codebook_visualization.png")
    print(f"   • output/codebook_usage_comparison.png")

    return True


def create_usage_visualization(usage_euc: Dict, usage_geo: Dict, save_path: str):
    """
    Crea una visualizzazione del confronto tra utilizzo euclideo e geodesico.
    """
    plt.figure(figsize=(15, 5))
    
    # Distribuzione utilizzo euclidea
    plt.subplot(1, 3, 1)
    euc_counts = usage_euc['usage_distribution']
    plt.hist(euc_counts, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Numero di assegnazioni')
    plt.ylabel('Numero di codici')
    plt.title('Distribuzione Utilizzo\n(Quantizzazione Euclidea)')
    plt.axvline(np.mean(euc_counts), color='red', linestyle='--', 
                label=f'Media: {np.mean(euc_counts):.1f}')
    plt.legend()
    
    # Distribuzione utilizzo geodesica
    plt.subplot(1, 3, 2)
    geo_counts = usage_geo['usage_distribution']
    plt.hist(geo_counts, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Numero di assegnazioni')
    plt.ylabel('Numero di codici')
    plt.title('Distribuzione Utilizzo\n(Quantizzazione Geodesica)')
    plt.axvline(np.mean(geo_counts), color='red', linestyle='--', 
                label=f'Media: {np.mean(geo_counts):.1f}')
    plt.legend()
    
    # Confronto diretto
    plt.subplot(1, 3, 3)
    indices = np.arange(len(euc_counts))
    width = 0.35
    
    plt.bar(indices - width/2, euc_counts, width, alpha=0.7, 
            color='blue', label='Euclidea')
    plt.bar(indices + width/2, geo_counts, width, alpha=0.7, 
            color='green', label='Geodesica')
    
    plt.xlabel('Indice Codice')
    plt.ylabel('Numero di assegnazioni')
    plt.title('Confronto Utilizzo\nper Codice')
    plt.legend()
    
    # Mostra solo i primi 50 codici per leggibilità
    if len(indices) > 50:
        plt.xlim(0, 50)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualizzazione utilizzo salvata: {save_path}")
    

def main():
    """
    Funzione principale.
    """
    print("AVVIO CREAZIONE CODEBOOK GEODESICO")
    print(f"Device: {device}")
    
    success = create_and_test_codebook()
    
    if success:
        print("\nPROCESSO COMPLETATO CON SUCCESSO!")
    else:
        print("\nERRORE DURANTE IL PROCESSO")


if __name__ == "__main__":
    main()
