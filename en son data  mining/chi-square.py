import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import os

def apply_chi_square(data, target_col, k, output_file):
    # Hedef klasörün varlığını kontrol et ve yoksa oluştur
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    chi_selector = SelectKBest(score_func=chi2, k=k)
    X_new = chi_selector.fit_transform(X, y)

    selected_features = pd.DataFrame(X_new, columns=data.columns[chi_selector.get_support(indices=True)])
    selected_features[target_col] = y

    selected_features.to_csv(output_file, index=False)
    print(f"Chi-square ile seçilen özellikler kaydedildi: {output_file}")

# Veri dosyaları ve klasörler
tfidf_files = [
    ('tfidf_unigram.csv', 'unigram_results/chi_square.csv'),
    ('tfidf_bigram.csv', 'bigram_results/chi_square.csv')
]

# Her dosya için Chi-Square uygula ve kaydet
for input_file, output_file in tfidf_files:
    data = pd.read_csv(input_file)
    apply_chi_square(data, 'Label', k=100, output_file=output_file)
