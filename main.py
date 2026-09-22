import pandas as pd

# Data load kar rahe hain
print("Loading dataset...")
matches_df = pd.read_csv('data/matches.csv')
deliveries_df = pd.read_csv('data/deliveries.csv')

# Check karte hain ki data sahi se load hua ya nahi
print("\n--- MATCHES DATA INFO ---")
print("Total Matches:", matches_df.shape)
print(matches_df.head(2))

print("\n--- DELIVERIES DATA INFO ---")
print("Total Deliveries:", deliveries_df.shape)
print(deliveries_df.head(2))




import pandas as pd

# --- STEP 1: Data Load Karna ---
print("Loading dataset...")
matches_df = pd.read_csv('data/matches.csv')
deliveries_df = pd.read_csv('data/deliveries.csv')

print("\n--- MATCHES DATA INFO ---")
print("Total Matches:", matches_df.shape)

print("\n--- DELIVERIES DATA INFO ---")
print("Total Deliveries:", deliveries_df.shape)


# --- STEP 2: Columns Check Karna ---
print("\n=== MATCHES COLUMNS ===")
print(matches_df.columns.tolist())

print("\n=== DELIVERIES COLUMNS ===")
print(deliveries_df.columns.tolist())


# --- STEP 3: Data Cleaning & Feature Selection ---
print("\n--- CLEANING DATA ---")

# 1. Sirf unhi matches ko rakhenge jinka koi winner decide hua ho (No Result ya Tie wale hata denge)
matches_df = matches_df.dropna(subset=['winner'])

# 2. Sirf zaroori columns ko select karte hain jo pre-match prediction ke liye chahiye
df_cleaned = matches_df[['team1', 'team2', 'city', 'venue', 'toss_winner', 'toss_decision', 'winner']].copy()

# Check karte hain ki cleaning ke baad kitne matches bache aur koi missing value toh nahi hai
print("Matches after cleaning:", df_cleaned.shape)
print("\nMissing values in each column:")
print(df_cleaned.isnull().sum())

# Cleaned data ke pehle 2 rows dekhte hain
print("\nSample Cleaned Data:")
print(df_cleaned.head(2))


# --- STEP 4: Handling Missing Values ---
print("\n--- HANDLING MISSING VALUES ---")

# Agar kuch matches mein city missing hai, toh hum use 'Unknown' ya venue se fill kar sakte hain
df_cleaned['city'] = df_cleaned['city'].fillna('Unknown')

# Check karte hain ki ab koi missing value toh nahi bachi
print("Missing values after cleaning:")
print(df_cleaned.isnull().sum())



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- STEP 5: Feature Encoding & Model Training ---
print("\n--- TRAINING MACHINE LEARNING MODEL ---")

# 1. Text data ko numbers mein convert karne ke liye LabelEncoder use karte hain
encoder = LabelEncoder()

df_encoded = df_cleaned.copy()
cat_columns = ['team1', 'team2', 'city', 'venue', 'toss_winner', 'toss_decision', 'winner']

for col in cat_columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])

# 2. Features (X) aur Target (y) alag karte hain
# Hum predict karenge winner, kiske base par? team1, team2, venue, toss_winner, toss_decision ke base par.
X = df_encoded[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']]
y = df_encoded['winner']

# 3. Data ko Train aur Test mein split karte hain (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Logistic Regression Model train karte hain
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Model ki Accuracy check karte hain
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Training Complete!")
print(f"Prediction Accuracy on Test Data: {accuracy * 100:.2f}%")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- STEP 5 (UPGRADED): Team Name Standardization & Random Forest Model ---
print("\n--- CLEANING TEAM NAMES & TRAINING RANDOM FOREST MODEL ---")

# 1. Team names ki spelling/updates ko standardize karte hain
def standardize_teams(team):
    if team == 'Delhi Daredevils':
        return 'Delhi Capitals'
    elif team == 'Kings XI Punjab':
        return 'Punjab Kings'
    elif team == 'Royal Challengers Bangalore':
        return 'Royal Challengers Bengaluru'
    elif team == 'Rising Pune Supergiants':
        return 'Rising Pune Supergiant'
    return team

df_cleaned['team1'] = df_cleaned['team1'].apply(standardize_teams)
df_cleaned['team2'] = df_cleaned['team2'].apply(standardize_teams)
df_cleaned['toss_winner'] = df_cleaned['toss_winner'].apply(standardize_teams)
df_cleaned['winner'] = df_cleaned['winner'].apply(standardize_teams)

# 2. Text data ko numbers mein encode karte hain
encoder = LabelEncoder()
df_encoded = df_cleaned.copy()
cat_columns = ['team1', 'team2', 'city', 'venue', 'toss_winner', 'toss_decision', 'winner']

for col in cat_columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))

# 3. Features aur Target define karte hain
X = df_encoded[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']]
y = df_encoded['winner']

# 4. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Random Forest Model Train karte hain (Yeh kafi powerful hota hai)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Accuracy check karte hain
y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest Model Training Complete!")
print(f"New Improved Prediction Accuracy: {rf_accuracy * 100:.2f}%")









import pickle

# Model aur Encoder ko save karte hain taaki website inhein use kar sake
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("\nModel and Encoder saved successfully as model.pkl and encoder.pkl!")