import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Performans değerlendirme fonksiyonu
def evaluate_model(data_file, method_name, results):
    if not os.path.exists(data_file):
        print(f"Dosya bulunamadı: {data_file}")
        return

    data = pd.read_csv(data_file)
    X = data.drop(columns=['Label'])
    y = data['Label']

    # Negatif değerleri sıfıra çevir
    X[X < 0] = 0

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree model eğitimi
    model = DecisionTreeClassifier(random_state=42)  # Karar ağacı modeli
    model.fit(X_train, y_train)

    # Test setinde tahmin
    y_pred = model.predict(X_test)

    # Performans metriklerini hesapla
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Sonuçları kaydet
    results.append({
        'Method': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# Sonuçları karşılaştırmak için fonksiyon
def compare_methods_decision_tree(result_folders, output_file):
    results = []
    for folder in result_folders:
        for method_file in ['chi_square.csv', 'pca.csv', 'variance_threshold.csv', 'l1_regularization.csv', 'tfidf.csv' ]:
            method_path = os.path.join(folder, method_file)
            method_name = f"{folder.split('_')[0]}_{method_file.split('.')[0]}"
            evaluate_model(method_path, method_name, results)

    # Sonuçları bir DataFrame'e dönüştür
    results_df = pd.DataFrame(results)

    # Excel dosyasına kaydet
    results_df.to_excel(output_file, index=False)
    print(f"Decision Tree Sonuçları kaydedildi: {output_file}")

# Unigram ve Bigram klasörleri
result_folders = ['unigram_results', 'bigram_results']

# Çıktı dosyası
output_excel_tree = 'decision_tree_results_comparison.xlsx'

# Yöntemleri karşılaştır ve Excel'e yaz
compare_methods_decision_tree(result_folders, output_excel_tree)
