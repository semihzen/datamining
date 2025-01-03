import pandas as pd
from sklearn.decomposition import PCA
import os

def apply_pca(data, target_col, n_components, output_file):
    # Hedef klasörün varlığını kontrol et ve yoksa oluştur
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df[target_col] = y

    pca_df.to_csv(output_file, index=False)
    print(f"PCA ile boyutu azaltılmış veri kaydedildi: {output_file}")

# Veri dosyaları ve klasörler
tfidf_files = [
    ('tfidf_unigram.csv', 'unigram_results/pca.csv'),
    ('tfidf_bigram.csv', 'bigram_results/pca.csv')
]

# Her dosya için PCA uygula ve kaydet
for input_file, output_file in tfidf_files:
    data = pd.read_csv(input_file)
    apply_pca(data, 'Label', n_components=50, output_file=output_file)
