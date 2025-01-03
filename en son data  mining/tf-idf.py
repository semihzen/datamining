import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Veri setini okuma
file_name = 'merged_dataset.txt'  # Veri setinizin adı
output_folder = './'  # Çıktılar aynı klasöre kaydedilecek

# Veri setini pandas DataFrame olarak yükleme
data = pd.read_csv(file_name, sep='\t', header=None, names=['Text', 'Label'])

# CSV TF-IDF Kaydetme Fonksiyonu
def apply_tfidf_to_csv(data, ngram_range, output_file):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(data['Text'])

    # TF-IDF sonuçlarını DataFrame'e dönüştürme
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Label sütununu ekleme
    tfidf_df['Label'] = data['Label']

    # CSV olarak kaydetme
    tfidf_df.to_csv(output_file, index=False)
    print(f"TF-IDF ({ngram_range}) sonuçları kaydedildi: {output_file}")

# Unigram için TF-IDF CSV kaydetme
apply_tfidf_to_csv(data, (1, 1), f"{output_folder}tfidf_unigram.csv")

# Bigram için TF-IDF CSV kaydetme
apply_tfidf_to_csv(data, (2, 2), f"{output_folder}tfidf_bigram.csv")
