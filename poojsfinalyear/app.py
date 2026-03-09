import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# --- File selectors for user and food datasets ---
st.sidebar.header('Dataset Selection')
import os
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
user_file = st.sidebar.selectbox('Select User Profile Dataset (INDB.csv)', csv_files, index=csv_files.index('INDB.csv') if 'INDB.csv' in csv_files else 0)
food_file = st.sidebar.selectbox('Select Food/Nutrition Dataset (diet_recommendations_dataset.csv)', csv_files, index=csv_files.index('diet_recommendations_dataset.csv') if 'diet_recommendations_dataset.csv' in csv_files else 0)


# Load user profile dataset
user_df = pd.read_csv(user_file)
user_df.columns = [col.strip().lower().replace(' ', '_') for col in user_df.columns]
st.write('Debug: user_df columns:', list(user_df.columns))
if 'age' not in user_df.columns:
	st.error(f"The selected user profile file '{user_file}' does not contain an 'age' column. Please select the correct file.")
	st.stop()

# Load diet recommendations/recipes dataset
try:
	diet_df = pd.read_csv(food_file)
except FileNotFoundError:
	st.error(f'{food_file} not found. Please add it to the project folder.')
	diet_df = None

st.title('Smart Nutritional Meal Planner & Grocery Assistant')

st.header('Enter Your Details')
name = st.text_input('Name')
age = st.number_input('Age', min_value=1, max_value=120, value=25)
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
weight = st.number_input('Weight (kg)', min_value=10, max_value=250, value=70)
# New inputs for disease and allergies
disease = st.selectbox('Disease Type', ['None', 'Diabetes', 'Hypertension', 'Obesity'])
allergies_input = st.text_input('Allergies (comma-separated)', 'None')
allergies = [a.strip() for a in allergies_input.split(',') if a.strip()]
plan_type = st.selectbox('Meal Plan Type', ['Daywise', 'Weekly'])

st.write('---')
# Calculate BMI
bmi = weight / ( (height/100) ** 2 ) if height > 0 else 0
st.write(f"**Calculated BMI:** {bmi:.1f}")

# Determine BMI category and recommended daily nutrients
if bmi < 18.5:
    bmi_cat = 'Underweight'
    rec_cal = 2500
    rec_vitA = 900
    rec_vitB = 1.2
    rec_vitC = 90
elif bmi < 25:
    bmi_cat = 'Normal weight'
    rec_cal = 2000
    rec_vitA = 800
    rec_vitB = 1.1
    rec_vitC = 75
elif bmi < 30:
    bmi_cat = 'Overweight'
    rec_cal = 1800
    rec_vitA = 700
    rec_vitB = 1.0
    rec_vitC = 65
else:
    bmi_cat = 'Obesity'
    rec_cal = 1600
    rec_vitA = 600
    rec_vitB = 0.9
    rec_vitC = 60
st.write(f"**BMI Category:** {bmi_cat}")
st.write(f"**Recommended Daily Calories:** {rec_cal} kcal")
st.write(f"**Recommended Daily Vitamin A:** {rec_vitA} µg")
st.write(f"**Recommended Daily Vitamin B1 (example):** {rec_vitB} mg")
st.write(f"**Recommended Daily Vitamin C:** {rec_vitC} mg")

st.write('---')
st.header('User Profile Dataset Preview')
st.dataframe(user_df.head())

if diet_df is not None:
	st.write('---')
	st.header('Diet Recommendation Dataset Preview')
	st.dataframe(diet_df.head())

	# --- Feature Extraction using TF-IDF and NMF (EnsTM + NMF) ---
	st.write('---')
	st.header('Food Feature Extraction (NMF Topics)')

	# Try to find a column with ingredients or description
	recipe_col = None
	for col in diet_df.columns:
		if 'recipe' in col.lower() or 'desc' in col.lower() or 'ingredient' in col.lower():
			recipe_col = col
			break

	if recipe_col:
		tfidf = TfidfVectorizer(stop_words='english', max_features=100)
		tfidf_matrix = tfidf.fit_transform(diet_df[recipe_col].astype(str))
		n_topics = min(5, tfidf_matrix.shape[0])
		nmf = NMF(n_components=n_topics, random_state=42)
		W = nmf.fit_transform(tfidf_matrix)
		H = nmf.components_
		feature_names = tfidf.get_feature_names_out()
		topics = []
		for topic_idx, topic in enumerate(H):
			top_terms = [feature_names[i] for i in topic.argsort()[:-6:-1]]
			topics.append(', '.join(top_terms))
		st.subheader('Extracted Food Topics:')
		for i, t in enumerate(topics):
			st.write(f"Topic {i+1}: {t}")
	else:
		st.warning('No recipe/ingredient/description column found for feature extraction in diet dataset.')

	# --- Simple Personalized Meal Plan & Grocery List ---
	st.write('---')
	st.header('Personalized Meal Plan & Grocery List')


	# Normalize column names to avoid KeyError due to whitespace/case

	user_df.columns = [col.strip().lower().replace(' ', '_') for col in user_df.columns]
	st.write('Debug: user_df columns:', list(user_df.columns))
	print('Debug: user_df columns:', list(user_df.columns))

	# Map user input to dietary recommendation
	# Find closest matching user in user_df
	user_row = user_df.loc[(user_df['age'] == age) &
						   (user_df['gender'].str.lower() == gender.lower()) &
						   (user_df['height_cm'] == height) &
						   (user_df['weight_kg'] == weight)]
	if user_row.empty:
		# If exact match not found, use only age/gender
		user_row = user_df.loc[(user_df['age'] == age) & (user_df['gender'].str.lower() == gender.lower())]
	if user_row.empty:
		st.warning('No exact user profile match found. Using default: Balanced diet.')
		diet_type = 'Balanced'
		restrictions = []
		allergies = []
	else:
		diet_type = user_row.iloc[0]['diet_recommendation'] if 'diet_recommendation' in user_row.columns else 'Balanced'
		restrictions = []
		if 'dietary_restrictions' in user_row.columns:
			restrictions = str(user_row.iloc[0]['dietary_restrictions']).split(',')
		allergies = []
		if 'allergies' in user_row.columns:
			allergies = str(user_row.iloc[0]['allergies']).split(',')

	st.write(f"**Diet Recommendation:** {diet_type}")
	st.write(f"**Dietary Restrictions:** {', '.join([r for r in restrictions if r and r.lower() != 'none']) or 'None'}")
	st.write(f"**Allergies:** {', '.join([a for a in allergies if a and a.lower() != 'none']) or 'None'}")

	# Filter foods based on restrictions/allergies
	food_df = diet_df.copy()
	# Filter out allergen-containing foods
	for allergen in allergies:
		if allergen and allergen.lower() != 'none':
			food_df = food_df[~food_df['diet_recommendation'].str.lower().str.contains(allergen.strip().lower())]
	# Filter for diet type (simple example: low_carb, balanced, low_sodium)
	# Determine diet_type from disease mapping
	disease_map = {'None':'Balanced', 'Diabetes':'Low_Sugar', 'Hypertension':'Low_Sodium', 'Obesity':'Low_Carb'}
	diet_type = disease_map.get(disease, 'Balanced')
	st.write(f"**Diet Type Based on Disease:** {diet_type}")
	if diet_type.lower() == 'low_carb':
		food_df = food_df.sort_values('carb_g').head(10)
	elif diet_type.lower() == 'low_sodium':
		# If sodium column exists
		sodium_col = None
		for col in food_df.columns:
			if 'sodium' in col.lower():
				sodium_col = col
				break
		if sodium_col:
			food_df = food_df.sort_values(sodium_col).head(10)
	else:
		# Balanced: pick top 10 foods by protein and fibre
		if 'protein_g' in food_df.columns and 'fibre_g' in food_df.columns:
			food_df = food_df.sort_values(['protein_g', 'fibre_g'], ascending=False).head(10)

	st.subheader('Recommended Foods for Your Meal Plan:')
	st.dataframe(food_df[['food_name', 'energy_kcal', 'carb_g', 'protein_g', 'fat_g', 'fibre_g']].reset_index(drop=True))

	# --- 7-Day Meal Plan (3 meals per day) ---
	st.write('---')
	st.subheader('7-Day Meal Plan (3 meals/day)')
	import numpy as _np
	days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
	meals = ['Breakfast','Lunch','Dinner']
	plan_rows = []
	for day in days:
		day_plan = {'Day': day}
		for meal in meals:
			# randomly select a recommended food
			if not food_df.empty:
				idx = _np.random.randint(len(food_df))
				day_plan[meal] = food_df.iloc[idx]['food_name']
			else:
				day_plan[meal] = 'N/A'
		plan_rows.append(day_plan)
	plan_df = pd.DataFrame(plan_rows)
	st.table(plan_df)

	# Instruction to user
	st.write('---')
	st.info('Please follow this 7-day meal plan and record your feedback. After one week, fill in your details again to refine further recommendations.')
else:
	st.warning('Diet recommendation dataset not loaded. Feature extraction and meal planning will be unavailable.')
