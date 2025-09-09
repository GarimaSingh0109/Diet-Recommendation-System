# app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Diet Recommendation System", page_icon="🥗", layout="wide")
st.title("🥗 Diet Recommendation System")

# -----------------------
# Helpers: metrics
# -----------------------
def bmi_value(weight_kg, height_cm):
    h_m = height_cm / 100.0
    return round(weight_kg / (h_m ** 2), 1)

def bmi_category(bmi, asian_adjusted=True):
    if asian_adjusted:
        # WHO Asian cutoffs
        if bmi < 18.5: return "Underweight"
        if bmi < 23: return "Healthy"
        if bmi < 25: return "Overweight (At risk)"
        if bmi < 30: return "Obese (Class I)"
        if bmi < 35: return "Obese (Class II)"
        return "Obese (Class III)"
    else:
        if bmi < 18.5: return "Underweight"
        if bmi < 25: return "Healthy"
        if bmi < 30: return "Overweight"
        if bmi < 35: return "Obese (Class I)"
        if bmi < 40: return "Obese (Class II)"
        return "Obese (Class III)"

def bmr_mifflin_st_jeor(sex, weight_kg, height_cm, age):
    if sex == "Male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

ACTIVITY_FACTORS = {
    "Sedentary (little/no exercise)": 1.2,
    "Light (1-3 days/wk)": 1.375,
    "Moderate (3-5 days/wk)": 1.55,
    "Very (6-7 days/wk)": 1.725,
    "Extra (physical job/training 2x)": 1.9,
}

# -----------------------
# Sidebar: inputs
# -----------------------
with st.sidebar:
    st.header("Profile")
    age = st.number_input("Age (years)", min_value=16, max_value=100, value=22, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    height = st.number_input("Height (cm)", min_value=130, max_value=220, value=170, step=1)
    activity = st.selectbox("Activity level", list(ACTIVITY_FACTORS.keys()))
    goal = st.selectbox("Goal", ["Lose weight", "Maintain", "Gain weight"])

    st.header("Preferences")
    is_veg = st.selectbox("Diet preference", ["Any", "Vegetarian", "Vegan"])
    avoid_list = st.text_input("Allergies/intolerances (comma-separated)", value="")

    st.header("Calorie strategy")
    # Let user choose adjustment percentage to avoid hardcoding a claim
    if goal == "Lose weight":
        pct = st.slider("Calorie deficit (%)", 5, 25, 15, step=1)
        adj = -pct / 100.0
    elif goal == "Gain weight":
        pct = st.slider("Calorie surplus (%)", 5, 20, 10, step=1)
        adj = pct / 100.0
    else:
        adj = 0.0

    st.header("Macros")
    preset = st.selectbox("Macro preset", ["Balanced (50/20/30)", "Higher protein (45/30/25)", "Custom"])
    if preset == "Balanced (50/20/30)":
        carbs_pct, prot_pct, fat_pct = 50, 20, 30
    elif preset == "Higher protein (45/30/25)":
        carbs_pct, prot_pct, fat_pct = 45, 30, 25
    else:
        carbs_pct = st.slider("Carbs %", 30, 60, 50, step=1)
        prot_pct = st.slider("Protein %", 15, 40, 25, step=1)
        fat_pct = 100 - carbs_pct - prot_pct
        st.caption(f"Fat % auto = {fat_pct}%")

# -----------------------
# Load foods data
# -----------------------
st.subheader("Upload Foods CSV")
st.write("Provide a CSV with columns like: name, calories_per_100g, protein_g_per_100g, fat_g_per_100g, carbs_g_per_100g, veg (True/False).")
file = st.file_uploader("Foods CSV", type=["csv"])

def normalize_foods(df):
    cols = {c.lower(): c for c in df.columns}
    def find(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    mapping = {
        "name": find("name","food","item"),
        "cal": find("calories_per_100g","calories","kcal","energy"),
        "p": find("protein_g_per_100g","protein","protein_g"),
        "f": find("fat_g_per_100g","fat","fat_g","total_fat"),
        "c": find("carbs_g_per_100g","carbs","carbohydrates","carbs_g"),
        "veg": find("veg","vegetarian","is_veg","vegan")
    }
    for k,v in mapping.items():
        if v is None and k != "veg":
            raise ValueError(f"Column for {k} not found; expected one of standard names.")
    out = pd.DataFrame({
        "name": df[mapping["name"]],
        "calories_per_100g": pd.to_numeric(df[mapping["cal"]], errors="coerce"),
        "protein_g_per_100g": pd.to_numeric(df[mapping["p"]], errors="coerce"),
        "fat_g_per_100g": pd.to_numeric(df[mapping["f"]], errors="coerce"),
        "carbs_g_per_100g": pd.to_numeric(df[mapping["c"]], errors="coerce"),
    })
    if mapping["veg"] is not None:
        veg_col = df[mapping["veg"]].astype(str).str.lower()
        out["veg"] = veg_col.isin(["true","1","yes","y","veg","vegetarian","vegan"])
    else:
        out["veg"] = np.nan
    out = out.dropna(subset=["calories_per_100g","protein_g_per_100g","fat_g_per_100g","carbs_g_per_100g"])
    out = out[out["calories_per_100g"] > 0]
    return out

foods = None
if file is not None:
    try:
        raw = pd.read_csv(file)
        foods = normalize_foods(raw)
        st.success(f"Loaded {len(foods)} foods")
        st.dataframe(foods.head())
    except Exception as e:
        st.error(f"Error reading foods CSV: {e}")

# -----------------------
# Compute targets
# -----------------------
col1, col2, col3, col4 = st.columns(4)
bmi = bmi_value(weight, height)
bmr = bmr_mifflin_st_jeor(sex, weight, height, age)
tdee = bmr * ACTIVITY_FACTORS[activity]
target_cals = int(round(tdee * (1 + adj)))

with col1: st.metric("BMI", f"{bmi}")
with col2: st.metric("BMI class", bmi_category(bmi, asian_adjusted=True))
with col3: st.metric("BMR (kcal)", f"{int(round(bmr))}")
with col4: st.metric("TDEE (kcal)", f"{int(round(tdee))}")

st.info(f"Target calories: {target_cals} kcal/day")

# Macro targets
carb_cals = target_cals * (carbs_pct/100.0)
prot_cals = target_cals * (prot_pct/100.0)
fat_cals  = target_cals * (fat_pct /100.0)
carb_g = int(round(carb_cals / 4))
prot_g = int(round(prot_cals / 4))
fat_g  = int(round(fat_cals / 9))

st.write(f"Macro targets: Carbs {carb_g} g • Protein {prot_g} g • Fat {fat_g} g")

# -----------------------
# Build a simple plan
# -----------------------
def filter_foods(df, is_veg, avoids):
    out = df.copy()
    if is_veg == "Vegetarian":
        out = out[(out["veg"].isna()) | (out["veg"] == True)]
    if is_veg == "Vegan":
        # If 'veg' is True it's vegetarian; without explicit vegan flag, keep veg and let user review
        out = out[(out["veg"].isna()) | (out["veg"] == True)]
    if avoids:
        terms = [a.strip().lower() for a in avoids.split(",") if a.strip()]
        if terms:
            mask = ~out["name"].str.lower().str.contains("|".join([pd.regex.escape(t) for t in terms]), na=False)
            out = out[mask]
    return out.reset_index(drop=True)

def meal_from_foods(df, meal_cal, n_items=3, prefer="balanced"):
    df = df.copy()
    if prefer == "protein":
        df["score"] = df["protein_g_per_100g"] / df["calories_per_100g"]
    elif prefer == "lowfat":
        df["score"] = -df["fat_g_per_100g"] / df["calories_per_100g"]
    else:
        df["score"] = (df["protein_g_per_100g"] + df["carbs_g_per_100g"]*0.5) / df["calories_per_100g"]
    picks = df.sort_values("score", ascending=False).head(max(10, n_items*3)).sample(n=min(n_items, len(df)), replace=False, random_state=42)
    per_item_cal = meal_cal / max(1, len(picks))
    rows = []
    for _, r in picks.iterrows():
        grams = max(50, min(350, (per_item_cal / r["calories_per_100g"]) * 100.0))
        kcal = r["calories_per_100g"] * grams / 100.0
        p = r["protein_g_per_100g"] * grams / 100.0
        f = r["fat_g_per_100g"] * grams / 100.0
        c = r["carbs_g_per_100g"] * grams / 100.0
        rows.append({"food": r["name"], "grams": int(round(grams)), "kcal": int(round(kcal)), "protein_g": int(round(p)), "fat_g": int(round(f)), "carbs_g": int(round(c))})
    meal_df = pd.DataFrame(rows)
    return meal_df

if foods is not None:
    filt = filter_foods(foods, is_veg, avoid_list)
    # meal calorie distribution
    b_cal = target_cals * 0.25
    l_cal = target_cals * 0.35
    d_cal = target_cals * 0.30
    s_cal = target_cals * 0.10

    prefer = "protein" if goal in ["Lose weight", "Gain weight"] else "balanced"

    st.subheader("Recommended Meal Plan")
    br = meal_from_foods(filt, b_cal, n_items=3, prefer=prefer)
    lu = meal_from_foods(filt, l_cal, n_items=3, prefer=prefer)
    di = meal_from_foods(filt, d_cal, n_items=3, prefer=prefer)
    sn = meal_from_foods(filt, s_cal, n_items=2, prefer="balanced")

    st.markdown("### Breakfast")
    st.dataframe(br)
    st.markdown("### Lunch")
    st.dataframe(lu)
    st.markdown("### Dinner")
    st.dataframe(di)
    st.markdown("### Snacks")
    st.dataframe(sn)

    total = pd.concat([br, lu, di, sn], ignore_index=True)
    totals = total[["kcal","protein_g","fat_g","carbs_g"]].sum()
    st.markdown("### Daily Totals")
    st.write({k:int(v) for k,v in totals.to_dict().items()})

    # Download
    def plan_to_md():
        def to_md(meal_name, df):
            lines = [f"#### {meal_name}"]
            for _, r in df.iterrows():
                lines.append(f"- {r['food']} — {r['grams']} g ({r['kcal']} kcal; P {r['protein_g']} g, F {r['fat_g']} g, C {r['carbs_g']} g)")
            return "\n".join(lines)
        parts = [
            f"# Diet Plan ({target_cals} kcal)",
            f"Macros: Carbs {carb_g} g • Protein {prot_g} g • Fat {fat_g} g",
            to_md("Breakfast", br),
            to_md("Lunch", lu),
            to_md("Dinner", di),
            to_md("Snacks", sn),
            f"\nTotal: {int(totals['kcal'])} kcal; P {int(totals['protein_g'])} g, F {int(totals['fat_g'])} g, C {int(totals['carbs_g'])} g",
        ]
        return "\n\n".join(parts)

    st.download_button("Download Plan (Markdown)", data=plan_to_md(), file_name="diet_plan.md", mime="text/markdown")
else:
    st.warning("Upload a foods CSV to generate a meal plan.")
