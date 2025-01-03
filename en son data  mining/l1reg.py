import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import os

def apply_l1_regularization(data, target_col, output_file):
    # Hedef klasörün varlığını kontrol et ve yoksa oluştur
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # L1 Regularization uygulama
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    l1_model.fit(X, y)

    selector = SelectFromModel(l1_model, prefit=True)
    X_new = selector.transform(X)

    selected_features = pd.DataFrame(X_new, columns=data.columns[selector.get_support(indices=True)])
    selected_features[target_col] = y

    selected_features.to_csv(output_file, index=False)
    print(f"L1 Regularization ile seçilen özellikler kaydedildi: {output_file}")

# Veri dosyaları ve klasörler
tfidf_files = [
    ('tfidf_unigram.csv', 'unigram_results/l1_regularization.csv'),
    ('tfidf_bigram.csv', 'bigram_results/l1_regularization.csv')
]

# Her dosya için L1 Regularization uygula ve kaydet
for input_file, output_file in tfidf_files:
    data = pd.read_csv(input_file)
    apply_l1_regularization(data, 'Label', output_file=output_file)
