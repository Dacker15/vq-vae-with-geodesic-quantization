import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_all_results(results_dir: str = "output/clustering_knn_graphs"):
    """
    Carica tutti i file di risultati dalla directory.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory risultati non trovata: {results_dir}")
        return []
    
    # Trova tutti i file .npz
    result_files = list(results_path.glob("clustering_*.npz"))
    
    if not result_files:
        print(f"Nessun file di risultati trovato in {results_dir}")
        return []
    
    print(f"Trovati {len(result_files)} file di risultati")
    
    all_results = []
    
    for file_path in sorted(result_files):
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Estrai configurazione dal nome del file
            filename = file_path.stem  # clustering_20_10_128
            parts = filename.split('_')[1:]  # ['20', '10', '128']
            
            if len(parts) >= 3:
                latent_dim, k_neighbors, n_clusters = map(int, parts[:3])
            else:
                print(f"Nome file non riconosciuto: {filename}")
                continue
            
            # Estrai metriche
            euclidean_metrics = data['euclidean_metrics'].item()
            geodesic_metrics = data['geodesic_metrics'].item()
            comparison_results = data['comparison_results'].item()
            
            result = {
                'file': file_path.name,
                'latent_dim': latent_dim,
                'k_neighbors': k_neighbors,
                'n_clusters': n_clusters,
                'euclidean_ari': euclidean_metrics['ari'],
                'euclidean_nmi': euclidean_metrics['nmi'],
                'geodesic_ari': geodesic_metrics['ari'],
                'geodesic_nmi': geodesic_metrics['nmi'],
                'ari_improvement': geodesic_metrics['ari'] - euclidean_metrics['ari'],
                'nmi_improvement': geodesic_metrics['nmi'] - euclidean_metrics['nmi'],
                'agreement_percentage': comparison_results['agreement_percentage'],
                'n_different_assignments': comparison_results['n_different_assignments'],
                'n_samples': len(data['true_labels'])
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Errore caricando {file_path}: {e}")
            continue

    print(f"Caricati {len(all_results)} risultati validi")
    return all_results

def create_summary_dataframe(results):
    """
    Crea un DataFrame pandas per analisi facili.
    """
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Aggiungi colonne derivate
    df['config'] = df.apply(lambda x: f"LD{x['latent_dim']}_k{x['k_neighbors']}_c{x['n_clusters']}", axis=1)
    
    # Classificazione miglioramento
    def classify_improvement(ari_imp):
        if ari_imp > 0.05:
            return "Significativo+"
        elif ari_imp > 0:
            return "Leggero+"
        elif ari_imp > -0.05:
            return "Comparabile"
        else:
            return "Peggiore"
    
    df['improvement_class'] = df['ari_improvement'].apply(classify_improvement)
    
    return df

def print_comprehensive_summary(df):
    """
    Stampa un riassunto completo dei risultati.
    """
    print("="*100)
    print("ANALISI COMPLETA RISULTATI CLUSTERING GEODESICO")
    print("="*100)
    
    if df.empty:
        print("Nessun dato disponibile per l'analisi")
        return
    
    n_experiments = len(df)
    print(f"\nSTATISTICHE GENERALI:")
    print(f"  • Esperimenti totali: {n_experiments}")
    print(f"  • Campioni per esperimento: {df['n_samples'].iloc[0] if not df.empty else 'N/A'}")
    print(f"  • Range ARI Euclideo: {df['euclidean_ari'].min():.3f} - {df['euclidean_ari'].max():.3f}")
    print(f"  • Range ARI Geodesico: {df['geodesic_ari'].min():.3f} - {df['geodesic_ari'].max():.3f}")
    
    # Statistiche miglioramenti
    print(f"\MIGLIORAMENTI:")
    print(f"  • ARI medio: {df['ari_improvement'].mean():+.3f} ± {df['ari_improvement'].std():.3f}")
    print(f"  • NMI medio: {df['nmi_improvement'].mean():+.3f} ± {df['nmi_improvement'].std():.3f}")
    print(f"  • Accordo medio: {df['agreement_percentage'].mean():.1f}% ± {df['agreement_percentage'].std():.1f}%")
    
    # Distribuzione classificazioni
    print(f"\nDISTRIBUZIONE MIGLIORAMENTI:")
    improvement_counts = df['improvement_class'].value_counts()
    for category, count in improvement_counts.items():
        percentage = (count / n_experiments) * 100
        print(f"  • {category}: {count}/{n_experiments} ({percentage:.1f}%)")
    
    # Top 5 migliori e peggiori
    print(f"\nTOP 5 MIGLIORI (ARI):")
    top_5 = df.nlargest(5, 'ari_improvement')
    for _, row in top_5.iterrows():
        print(f"  {row['config']}: Δ{row['ari_improvement']:+.3f} (ARI: {row['euclidean_ari']:.3f}→{row['geodesic_ari']:.3f})")
    
    print(f"\n⬇TOP 5 PEGGIORI (ARI):")
    bottom_5 = df.nsmallest(5, 'ari_improvement')
    for _, row in bottom_5.iterrows():
        print(f"  {row['config']}: Δ{row['ari_improvement']:+.3f} (ARI: {row['euclidean_ari']:.3f}→{row['geodesic_ari']:.3f})")

def analyze_by_parameters(df):
    """
    Analizza i risultati raggruppati per parametri.
    """
    if df.empty:
        return
        
    print(f"\n" + "="*80)
    print("ANALISI PER PARAMETRI")
    print("="*80)
    
    # Analisi per dimensione latente
    print(f"\nPER DIMENSIONE LATENTE:")
    by_latent = df.groupby('latent_dim').agg({
        'ari_improvement': ['mean', 'std', 'count'],
        'nmi_improvement': ['mean', 'std'],
        'agreement_percentage': 'mean'
    }).round(3)
    
    for ld in sorted(df['latent_dim'].unique()):
        subset = df[df['latent_dim'] == ld]
        print(f"  LD={ld}: ARI Δ{subset['ari_improvement'].mean():+.3f}±{subset['ari_improvement'].std():.3f}, "
              f"NMI Δ{subset['nmi_improvement'].mean():+.3f}±{subset['nmi_improvement'].std():.3f}, "
              f"Accordo {subset['agreement_percentage'].mean():.1f}%")
    
    # Analisi per k-neighbors
    print(f"\nPER K-NEIGHBORS:")
    for k in sorted(df['k_neighbors'].unique()):
        subset = df[df['k_neighbors'] == k]
        print(f"  k={k}: ARI Δ{subset['ari_improvement'].mean():+.3f}±{subset['ari_improvement'].std():.3f}, "
              f"NMI Δ{subset['nmi_improvement'].mean():+.3f}±{subset['nmi_improvement'].std():.3f}, "
              f"Accordo {subset['agreement_percentage'].mean():.1f}%")
    
    # Analisi per n-clusters
    print(f"\nPER N-CLUSTERS:")
    for nc in sorted(df['n_clusters'].unique()):
        subset = df[df['n_clusters'] == nc]
        print(f"  clusters={nc}: ARI Δ{subset['ari_improvement'].mean():+.3f}±{subset['ari_improvement'].std():.3f}, "
              f"NMI Δ{subset['nmi_improvement'].mean():+.3f}±{subset['nmi_improvement'].std():.3f}, "
              f"Accordo {subset['agreement_percentage'].mean():.1f}%")

def create_visualizations(df, output_dir="output/clustering_knn_graphs"):
    """
    Crea visualizzazioni dei risultati.
    """
    if df.empty:
        return
        
    print(f"\nGenerazione visualizzazioni...")
    
    # Setup style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Heatmap dei miglioramenti ARI
    plt.figure(figsize=(15, 10))
    
    # Crea pivot table per heatmap
    pivot_data = []
    for ld in sorted(df['latent_dim'].unique()):
        for k in sorted(df['k_neighbors'].unique()):
            row = {'latent_dim': ld, 'k_neighbors': k}
            for nc in sorted(df['n_clusters'].unique()):
                subset = df[(df['latent_dim'] == ld) & 
                           (df['k_neighbors'] == k) & 
                           (df['n_clusters'] == nc)]
                if not subset.empty:
                    row[f'clusters_{nc}'] = subset['ari_improvement'].iloc[0]
                else:
                    row[f'clusters_{nc}'] = np.nan
            pivot_data.append(row)
    
    if pivot_data:
        pivot_df = pd.DataFrame(pivot_data)
        pivot_df = pivot_df.set_index(['latent_dim', 'k_neighbors'])
        
        plt.subplot(2, 2, 1)
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'ARI Improvement'})
        plt.title('Miglioramento ARI per Configurazione')
        plt.ylabel('(Latent Dim, k-neighbors)')
    
    # 2. Distribuzione miglioramenti
    plt.subplot(2, 2, 2)
    df['ari_improvement'].hist(bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Nessun miglioramento')
    plt.xlabel('Miglioramento ARI')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione Miglioramenti ARI')
    plt.legend()
    
    # 3. Scatter plot ARI vs NMI improvements
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(df['ari_improvement'], df['nmi_improvement'], 
                         c=df['latent_dim'], s=50, alpha=0.7, cmap='viridis')
    plt.xlabel('Miglioramento ARI')
    plt.ylabel('Miglioramento NMI')
    plt.title('ARI vs NMI Improvement')
    plt.colorbar(scatter, label='Latent Dim')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 4. Box plot per parametri
    plt.subplot(2, 2, 4)
    df_melted = df.melt(id_vars=['latent_dim'], 
                       value_vars=['ari_improvement'], 
                       var_name='metric', value_name='improvement')
    sns.boxplot(data=df_melted, x='latent_dim', y='improvement')
    plt.xlabel('Dimensione Latente')
    plt.ylabel('Miglioramento ARI')
    plt.title('Miglioramento per Dimensione Latente')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Grafico dettagliato per k-neighbors
    plt.figure(figsize=(12, 8))
    
    for ld in sorted(df['latent_dim'].unique()):
        subset = df[df['latent_dim'] == ld]
        k_means = subset.groupby('k_neighbors')['ari_improvement'].mean()
        k_stds = subset.groupby('k_neighbors')['ari_improvement'].std()
        
        plt.errorbar(k_means.index, k_means.values, yerr=k_stds.values, 
                    marker='o', label=f'Latent Dim {ld}', linewidth=2, markersize=6)
    
    plt.xlabel('k-neighbors')
    plt.ylabel('Miglioramento ARI Medio')
    plt.title('Effetto di k-neighbors sul Miglioramento ARI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.savefig(f"{output_dir}/k_neighbors_effect.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualizzazioni salvate in {output_dir}")

def main():
    """
    Funzione principale per l'analisi completa.
    """
    print("ANALISI COMPLETA RISULTATI CLUSTERING GEODESICO")
    print("="*60)
    
    # Carica tutti i risultati
    results = load_all_results()
    
    if not results:
        print("Nessun risultato trovato. Esegui prima gli esperimenti!")
        return
    
    # Crea DataFrame
    df = create_summary_dataframe(results)
    
    # Analisi completa
    print_comprehensive_summary(df)
    analyze_by_parameters(df)
    
    # Crea visualizzazioni
    create_visualizations(df)
    
    # Salva summary in CSV
    output_path = "output/clustering_knn_graphs/results_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\nRiassunto salvato in: {output_path}")
    
    # Raccomandazioni finali
    print(f"\nRACCOMANDAZIONI:")

    if not df.empty:
        best_config = df.loc[df['ari_improvement'].idxmax()]
        worst_config = df.loc[df['ari_improvement'].idxmin()]
        
        print(f"  Migliore configurazione: {best_config['config']} (Δ{best_config['ari_improvement']:+.3f})")
        print(f"  Peggiore configurazione: {worst_config['config']} (Δ{worst_config['ari_improvement']:+.3f})")
        
        # Percentuale di miglioramenti
        improvements = (df['ari_improvement'] > 0).sum()
        total = len(df)
        print(f"  Configurazioni con miglioramento: {improvements}/{total} ({improvements/total*100:.1f}%)")

if __name__ == "__main__":
    main()
