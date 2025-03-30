import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
import re


df = pd.read_csv('venv/Pokemon.csv')


print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())

df['is_mega'] = df['Name'].str.contains('Mega', case=False).astype(int)

print("\nClass distribution:")
print(df['is_mega'].value_counts())
print(f"Percentage of Mega Evolutions: {df['is_mega'].mean() * 100:.2f}%")

statistical_features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']


for col in statistical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

X = df[statistical_features]
y = df['is_mega']
pokemon_names = df['Name']

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, pokemon_names, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
y_pred_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

results_df = pd.DataFrame({
    'Pokemon': names_test,
    'Actual': y_test,
    'Predicted': y_pred,
    'Probability': y_pred_prob
})

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Regular', 'Mega'], 
            yticklabels=['Regular', 'Mega'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# ROC curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

# Precision-Recall curve
plt.figure(figsize=(10, 8))
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig('precision_recall_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'Feature': statistical_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Mega Evolution Classification')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()


all_features_scaled = scaler.transform(df[statistical_features])
all_predictions = rf_model.predict(all_features_scaled)

final_output = pd.DataFrame({
    'Pokemon': df['Name'],
    'Mega_Evolution': ['Yes' if pred == 1 else 'No' for pred in all_predictions]
})

# Save to CSV
final_output.to_csv('mega_evolution_predictions.csv', index=False)
print("\nFinal output saved to 'mega_evolution_predictions.csv'")

print("\nSample predictions:")
print(final_output.head(10))

mega_pokemon = df[df['is_mega'] == 1]['Name'].tolist()
print(f"\nActual Mega Evolutions in dataset ({len(mega_pokemon)}):")
print(mega_pokemon)

mega_predictions = final_output[final_output['Mega_Evolution'] == 'Yes']['Pokemon'].tolist()
print(f"\nPredicted Mega Evolutions ({len(mega_predictions)}):")
print(mega_predictions[:20]) 

# Calculate and display metrics
correctly_identified = len(set(mega_pokemon) & set(mega_predictions))
if len(mega_pokemon) > 0:
    recall = correctly_identified / len(mega_pokemon)
    print(f"\nModel recall for Mega Evolutions: {recall:.2f}")

if len(mega_predictions) > 0:
    precision = correctly_identified / len(mega_predictions)
    print(f"Model precision for Mega Evolutions: {precision:.2f}")