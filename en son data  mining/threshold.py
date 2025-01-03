import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import os

def apply_variance_threshold(data, target_col, threshold, output_file):
    # Hedef klasörün varlığını kontrol et ve yoksa oluştur
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Variance Threshold uygulama
    selector = VarianceThreshold(threshold=threshold)
    X_new = selector.fit_transform(X)

    selected_features = pd.DataFrame(X_new, columns=data.columns[selector.get_support(indices=True)])
    selected_features[target_col] = y

    selected_features.to_csv(output_file, index=False)
    print(f"Variance Threshold ile seçilen özellikler kaydedildi: {output_file}")

# Veri dosyaları ve klasörler
tfidf_files = [
    ('tfidf_unigram.csv', 'unigram_results/variance_threshold.csv'),
    ('tfidf_bigram.csv', 'bigram_results/variance_threshold.csv')
]

# Her dosya için Variance Threshold uygula ve kaydet
for input_file, output_file in tfidf_files:
    data = pd.read_csv(input_file)
    apply_variance_threshold(data, 'Label', threshold=0.001, output_file=output_file)
