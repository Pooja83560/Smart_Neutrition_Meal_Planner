from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import random
import re
import os
import glob
import requests
from datetime import datetime, timedelta

app = FastAPI()
# Use paths relative to this file so imports work regardless of CWD
base_dir = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

def get_youtube_recipe_link(recipe_name):
    """Generate YouTube search URL for recipe"""
    query = recipe_name.replace(' ', '+')
    return f"https://www.youtube.com/results?search_query={query}+recipe+indian+cooking"

def get_youtube_recipe_link(recipe_name):
    """Generate YouTube search URL for recipe"""
    query = recipe_name.replace(' ', '+')
    return f"https://www.youtube.com/results?search_query={query}+recipe+indian+cooking"

# Load datasets using paths relative to this file so CWD doesn't matter
try:
    diet_df = pd.read_csv(os.path.join(base_dir, "INDB.csv"))
except Exception:
    diet_df = pd.DataFrame()

# Optionally load a profile-based suggestions dataset if present
profile_df = pd.DataFrame()
# Candidate profile/suggestions files (look in package folder)
candidate_names = [
    "Food_and_Nutrition__.csv",  # your uploaded file
    "profile_meal_suggestions.csv",
    "meal_suggestions.csv",
    "dietary_profile_suggestions.csv",
    "meal_recommendations.csv",
]
for fname in candidate_names:
    path = os.path.join(base_dir, fname)
    if os.path.exists(path):
        try:
            profile_df = pd.read_csv(path)
            break
        except Exception:
            profile_df = pd.DataFrame()

# Generic auto-discovery: pick any CSV with required suggestion columns
if profile_df.empty:
    required_cols = {
        'Dietary Preference', 'Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion'
    }
    for path in glob.glob(os.path.join(base_dir, "*.csv")):
        try:
            tmp = pd.read_csv(path)
            if required_cols.issubset(set(tmp.columns)):
                profile_df = tmp
                break
        except Exception:
            continue

# Disease to diet mapping
disease_map = {
    'None': 'Balanced',
    'Diabetes': 'Low_Sugar',
    'Hypertension': 'Low_Sodium',
    'Obesity': 'Low_Carb'
}

# Indian cuisine API integration
def get_indian_meal_suggestions(diet_pref, health_condition, calories=2000, meal_type='main course'):
    """Get Indian meal suggestions from Spoonacular API or fallback to local Indian foods"""
    suggestions = []
    
    # Try Spoonacular API first (requires API key)
    spoonacular_key = os.getenv('SPOONACULAR_API_KEY')  # Set this in environment
    if spoonacular_key:
        try:
            url = "https://api.spoonacular.com/recipes/complexSearch"
            params = {
                'apiKey': spoonacular_key,
                'cuisine': 'indian',
                'diet': 'vegetarian' if diet_pref.lower() in ['veg', 'vegetarian'] else None,
                'maxCalories': calories,
                'number': 10,
                'addRecipeInformation': True
            }
            if diet_pref.lower() == 'vegan':
                params['diet'] = 'vegan'
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for recipe in data.get('results', [])[:5]:
                    suggestions.append(recipe.get('title', 'Indian Recipe'))
        except Exception:
            pass  # Fallback to local suggestions
    
    # Fallback to curated Indian meal suggestions
    if not suggestions:
        indian_meals = {
            'vegetarian': {
                'breakfast': ['Poha with vegetables', 'Upma with sambar', 'Idli with coconut chutney', 
                             'Dosa with potato curry', 'Paratha with curd', 'Aloo paratha with pickle'],
                'lunch': ['Dal rice with vegetable curry', 'Rajma with roti', 'Chole with bhature', 
                         'Sambar rice with papad', 'Biryani with raita', 'Paneer curry with naan'],
                'dinner': ['Khichdi with ghee', 'Dal tadka with roti', 'Vegetable curry with rice', 
                          'Paneer butter masala with naan', 'Mixed dal with chapati', 'Aloo gobi with roti']
            },
            'vegan': {
                'breakfast': ['Poha with peanuts', 'Upma with vegetables', 'Idli with sambar', 
                             'Dosa with chutney', 'Ragi porridge', 'Oats upma'],
                'lunch': ['Dal rice with pickle', 'Rajma with roti', 'Chana masala with rice', 
                         'Sambar rice', 'Vegetable biryani', 'Mixed vegetable curry with roti'],
                'dinner': ['Khichdi with turmeric', 'Dal with chapati', 'Vegetable curry with rice', 
                          'Aloo curry with roti', 'Mixed dal with bread', 'Sabzi with roti']
            },
            'omnivore': {
                'breakfast': ['Egg curry with paratha', 'Chicken sandwich', 'Omelette with bread', 
                             'Boiled eggs with upma', 'Egg bhurji with roti', 'Masala omelette'],
                'lunch': ['Chicken curry with rice', 'Mutton biryani', 'Fish curry with roti', 
                         'Egg curry with dal rice', 'Chicken biryani', 'Keema with naan'],
                'dinner': ['Dal chicken with rice', 'Fish fry with roti', 'Chicken curry with chapati', 
                          'Egg curry with bread', 'Mutton curry with rice', 'Chicken masala with naan']
            }
        }
        
        pref_key = 'vegetarian' if diet_pref.lower() in ['veg', 'vegetarian'] else \
                  'vegan' if diet_pref.lower() == 'vegan' else 'omnivore'
        
        meal_category = 'lunch'  # default
        if meal_type.lower() in ['breakfast', 'lunch', 'dinner']:
            meal_category = meal_type.lower()
            
        suggestions = indian_meals.get(pref_key, {}).get(meal_category, [])
    
    return suggestions[:5] if suggestions else ['Mixed dal with rice', 'Vegetable curry with roti']

@app.get('/', response_class=HTMLResponse)
async def form(request: Request):
    diseases = ['None', 'Diabetes', 'Hypertension', 'Obesity']
    return templates.TemplateResponse("index.html", {"request": request, "diseases": diseases})

@app.get('/meal-prep', response_class=HTMLResponse)
async def meal_prep(request: Request):
    return templates.TemplateResponse("meal_prep.html", {"request": request})

@app.get('/recipes', response_class=HTMLResponse)
async def recipes(request: Request):
    # Sample Indian recipes with YouTube links
    recipe_list = [
        {"name": "Butter Chicken", "time": "45 mins", "difficulty": "Medium", "calories": 520},
        {"name": "Biryani", "time": "90 mins", "difficulty": "Hard", "calories": 650},
        {"name": "Dal Tadka", "time": "30 mins", "difficulty": "Easy", "calories": 280},
        {"name": "Paneer Tikka Masala", "time": "40 mins", "difficulty": "Medium", "calories": 420},
        {"name": "Chole Bhature", "time": "60 mins", "difficulty": "Medium", "calories": 580},
        {"name": "Rajma Chawal", "time": "50 mins", "difficulty": "Easy", "calories": 380},
        {"name": "Samosa", "time": "45 mins", "difficulty": "Medium", "calories": 150},
        {"name": "Palak Paneer", "time": "35 mins", "difficulty": "Easy", "calories": 320},
        {"name": "Tandoori Chicken", "time": "2 hours", "difficulty": "Medium", "calories": 450},
        {"name": "Masala Dosa", "time": "8 hours", "difficulty": "Hard", "calories": 250}
    ]
    
    # Add YouTube links to each recipe
    for recipe in recipe_list:
        recipe["youtube_link"] = get_youtube_recipe_link(recipe["name"])
    
    return templates.TemplateResponse("recipes.html", {"request": request, "recipes": recipe_list})

@app.get('/progress', response_class=HTMLResponse)
async def progress(request: Request):
    # Simple progress page without complex tracking
    return templates.TemplateResponse("progress.html", {"request": request})

@app.get('/shopping', response_class=HTMLResponse)
async def shopping(request: Request):
    return templates.TemplateResponse("shopping.html", {"request": request})

@app.post('/plan', response_class=HTMLResponse)
async def plan(request: Request,
               name: str = Form(...),
               age: int = Form(...),
               gender: str = Form(...),
               weight: float = Form(...),
               height: float = Form(...),
               disease: str = Form(...),
               activity_level: str = Form('Moderately Active'),
               allergies: str = Form('None'),
               plan_type: str = Form('Weekly'),
               diet_pref: str = Form('Non-Veg')):
    # Process inputs
    bmi = weight / ((height/100)**2) if height > 0 else 0
    # Recommend nutrients based on BMI
    if bmi < 18.5:
        bmi_cat = 'Underweight'; rec_cal = 2500; rec_a = 900; rec_b = 1.2; rec_c = 90
    elif bmi < 25:
        bmi_cat = 'Normal'; rec_cal = 2000; rec_a = 800; rec_b = 1.1; rec_c = 75
    elif bmi < 30:
        bmi_cat = 'Overweight'; rec_cal = 1800; rec_a = 700; rec_b = 1.0; rec_c = 65
    else:
        bmi_cat = 'Obese'; rec_cal = 1600; rec_a = 600; rec_b = 0.9; rec_c = 60

    # Ensure diet data is loaded and has required columns
    if diet_df.empty:
        # Nutrition database not loaded — show expected path
            expected = os.path.join(base_dir, "INDB.csv")
            raise HTTPException(status_code=500, detail=f"Nutrition database (INDB.csv) is not loaded. Please add '{expected}' to the project folder.")
    if 'food_name' not in diet_df.columns:
        # Provide debug info about available columns
        available = ', '.join(diet_df.columns.tolist())
        raise HTTPException(status_code=500, detail=f"Missing 'food_name' column. Available columns: {available}")
    # Determine diet type
    diet_type = disease_map.get(disease, 'Balanced')
    # First, try to use profile suggestions dataset if available
    plan = []
    used_profile = False
    show_snack = False
    if not profile_df.empty and all(
        col in profile_df.columns for col in [
            'Dietary Preference', 'Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion'
        ]
    ):
        dfp = profile_df.copy()
        # Normalize columns for matching
        for c in dfp.columns:
            if dfp[c].dtype == object:
                dfp[c] = dfp[c].astype(str)
        # Filter by dietary preference (map Non-Veg -> Omnivore)
        pref_in = (diet_pref or 'Non-Veg').strip().lower()
        if pref_in in ['non-veg', 'non veg']:
            pref_key = 'omnivore'
        elif pref_in in ['veg', 'vegetarian']:
            pref_key = 'vegetarian'
        elif pref_in == 'vegan':
            pref_key = 'vegan'
        else:
            pref_key = 'omnivore'  # default fallback
        
        # Debug: Check what we're filtering for
        print(f"Looking for dietary preference: {pref_key} (from user input: {pref_in})")
        print(f"Available preferences in dataset: {dfp['Dietary Preference'].unique()}")
        
        dfp = dfp[dfp['Dietary Preference'].str.lower() == pref_key]
        # Optional filters if columns exist
        if 'Gender' in dfp.columns:
            dfp = dfp[(dfp['Gender'].str.lower() == gender.lower()) | (~dfp['Gender'].notna())]
        if 'Disease' in dfp.columns and disease and disease != 'None':
            dfp = dfp[dfp['Disease'].str.contains(re.escape(disease), case=False, na=False)]
        if 'Activity Level' in dfp.columns and activity_level:
            dfp = dfp[dfp['Activity Level'].str.lower() == activity_level.lower()]
        # Pick best matches by nearest age if Ages/Age present
        if 'Ages' in dfp.columns:
            try:
                dfp['_age_diff'] = (pd.to_numeric(dfp['Ages'], errors='coerce') - age).abs()
            except Exception:
                dfp['_age_diff'] = np.inf
        elif 'Age' in dfp.columns:
            try:
                dfp['_age_diff'] = (pd.to_numeric(dfp['Age'], errors='coerce') - age).abs()
            except Exception:
                dfp['_age_diff'] = np.inf
        else:
            dfp['_age_diff'] = 0

        if not dfp.empty:
            # Build a week by sampling rows (with replacement if needed)
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            meals = ['Breakfast', 'Lunch', 'Dinner']
            # Prefer closest age matches first
            dfp_sorted = dfp.sort_values('_age_diff')
            rows = dfp_sorted.head(14)  # a buffer to vary meals across week
            # Cycle through these to fill 7 days
            idx = 0
            for d in days:
                r = rows.iloc[idx % len(rows)]
                entry = {
                    'day': d,
                    'Breakfast': r.get('Breakfast Suggestion', 'N/A'),
                    'Lunch': r.get('Lunch Suggestion', 'N/A'),
                    'Dinner': r.get('Dinner Suggestion', 'N/A'),
                }
                if 'Snack Suggestion' in rows.columns:
                    entry['Snack'] = r.get('Snack Suggestion', 'N/A')
                plan.append(entry)
                idx += 1
            used_profile = True
            show_snack = 'Snack Suggestion' in rows.columns

    # If profile suggestions not used, use Indian cuisine API or curated Indian meals
    if not used_profile:
        # Generate Indian meal plan using API or local suggestions
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plan = []
        grocery = []
        
        # Map dietary preference for Indian cuisine - ensure strict filtering
        pref_in = (diet_pref or 'Non-Veg').strip().lower()
        if pref_in in ['veg', 'vegetarian']:
            pref_key = 'vegetarian'
        elif pref_in == 'vegan':
            pref_key = 'vegan'
        else:
            pref_key = 'omnivore'
        
        print(f"Using Indian cuisine preference: {pref_key} for user selection: {pref_in}")
        
        for day in days:
            entry = {'day': day}
            # Get suggestions for each meal type
            breakfast_options = get_indian_meal_suggestions(diet_pref, disease, rec_cal//3, 'breakfast')
            lunch_options = get_indian_meal_suggestions(diet_pref, disease, rec_cal//2, 'lunch')  
            dinner_options = get_indian_meal_suggestions(diet_pref, disease, rec_cal//3, 'dinner')
            
            entry['Breakfast'] = random.choice(breakfast_options) if breakfast_options else 'Indian breakfast'
            entry['Lunch'] = random.choice(lunch_options) if lunch_options else 'Indian lunch'
            entry['Dinner'] = random.choice(dinner_options) if dinner_options else 'Indian dinner'
            
            plan.append(entry)
            
        # Generate nutrition summary instead of grocery list
        nutrition_summary = {
            'total_meals': len(plan) * 3,
            'dietary_preference': diet_pref,
            'health_focus': diet_type,
            'weekly_variety': len(set([row['Breakfast'] for row in plan] + [row['Lunch'] for row in plan] + [row['Dinner'] for row in plan]))
        }
    else:
        # Grocery list derived from unique dishes in the plan (profile-based)
        items = [row['Breakfast'] for row in plan] + [row['Lunch'] for row in plan] + [row['Dinner'] for row in plan]
        if show_snack:
            items += [row.get('Snack') for row in plan]
        grocery = list(dict.fromkeys([x for x in items if x and x != 'N/A']))

    return templates.TemplateResponse('result.html', {
        'request': request,
        'name': name,
        'age': age,
        'gender': gender,
        'bmi': f"{bmi:.1f}",
        'bmi_cat': bmi_cat,
        'rec_cal': rec_cal,
        'rec_a': rec_a,
        'rec_b': rec_b,
        'rec_c': rec_c,
        'diet_type': diet_type,
        'plan': plan,
        'nutrition_summary': nutrition_summary if not used_profile else {
            'total_meals': len(plan) * 3,
            'dietary_preference': diet_pref,
            'health_focus': diet_type,
            'weekly_variety': len(set([row['Breakfast'] for row in plan] + [row['Lunch'] for row in plan] + [row['Dinner'] for row in plan]))
        },
        'plan_type': plan_type,
        'diet_pref': diet_pref,
        'show_snack': show_snack
    })
