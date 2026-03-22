import pandas as pd 
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS



def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words =[w for w in words if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words)
    
df_fake = pd.read_csv("training/Fake.csv")
df_real = pd.read_csv("training/True.csv") 

df_fake['label']= 0
df_real['label']= 1

df = pd.concat([df_fake, df_real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df[['text', 'label']]

print("Cleaning text data....it may take a minute.")
df['text'] = df['text'].apply(clean_text)

df.to_csv('training/processed_data.csv', index=False)
print(f"Done. Dataset shape: {df.shape}")
print(df['label'].value_counts())


