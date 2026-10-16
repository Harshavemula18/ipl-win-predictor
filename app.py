
import base64
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================================
# 1. PAGE CONFIGURATION
# =========================================
ROOT = Path(__file__).parent

st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
)

# =========================================
# 2. IPL TEAM DETAILS
# =========================================
TEAM_DETAILS = [
    ("CSK", "Chennai Super Kings", "#ffcc22", "🦁"),
    ("MI", "Mumbai Indians", "#2787ff", "🌊"),
    ("RCB", "Royal Challengers Bengaluru", "#ff4057", "👑"),
    ("KKR", "Kolkata Knight Riders", "#b68aff", "⚡"),
    ("SRH", "Sunrisers Hyderabad", "#ff812d", "🦅"),
    ("RR", "Rajasthan Royals", "#ff70bb", "👑"),
    ("DC", "Delhi Capitals", "#4e9cff", "🏏"),
    ("PBKS", "Punjab Kings", "#ff5c66", "🐯"),
    ("LSG", "Lucknow Super Giants", "#46d7d7", "🦅"),
    ("GT", "Gujarat Titans", "#b4c9ed", "🔷"),
]

# =========================================
# 3. STADIUM BACKGROUND
# =========================================
image_path = ROOT / "assets" / "ipl_stadium.png"

if image_path.exists():
    encoded = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    background = f"""
    background-image:
        linear-gradient(
            rgba(2, 8, 25, .76),
            rgba(2, 8, 25, .90)
        ),
        url("data:image/png;base64,{encoded}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    """
else:
    background = """
    background:
        radial-gradient(ellipse at top, #123a70, transparent 60%),
        linear-gradient(135deg, #020817, #160d30);
    """

# =========================================
# 4. WEBSITE DESIGN
# =========================================
st.markdown(
    f"""
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap'
    );

    .stApp {{
        {background}
        color: white;
        font-family: Inter, sans-serif;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}

    .block-container {{
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}

    .hero {{
        text-align: center;
        padding: 25px 0 35px;
    }}

    .eyebrow {{
        color: #61d1ff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 4px;
    }}

    .hero h1 {{
        color: white;
        font-size: clamp(36px, 5vw, 66px);
        font-weight: 900;
        margin: 15px 0;
    }}

    .hero h1 span {{
        color: #ffca3d;
    }}

    .hero p {{
        color: #d0dff5;
        font-size: 16px;
    }}

    .panel {{
        background: rgba(4, 18, 42, .96);
        border: 1px solid #23649a;
        border-radius: 20px;
        padding: 24px;
        margin: 14px 0;
        box-shadow: 0 12px 40px rgba(0,0,0,.25);
    }}

    .panel h2 {{
        color: white;
        margin: 0 0 8px;
    }}

    .muted {{
        color: #b8cce6;
        font-size: 13px;
    }}

    .stSelectbox label {{
        color: #e5efff !important;
        font-weight: 700 !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: #102743;
        border: 1px solid #31577b;
        border-radius: 10px;
    }}

    .stButton > button {{
        width: 100%;
        min-height: 52px;
        border: 0;
        border-radius: 11px;
        color: white;
        font-size: 16px;
        font-weight: 800;
        background: linear-gradient(95deg, #ff287c, #8b36ff);
    }}

    .stButton > button:hover {{
        border: 1px solid white;
        color: white;
        filter: brightness(1.15);
    }}

    .result-card {{
        background: rgba(3, 16, 38, .97);
        border: 1px solid #168cff;
        border-radius: 17px;
        padding: 22px 12px;
        text-align: center;
        margin: 12px 0;
        min-height: 150px;
    }}

    .team-name {{
        color: #e7f0ff;
        font-size: 15px;
        font-weight: 700;
        min-height: 42px;
    }}

    .prob {{
        font-size: clamp(28px, 3vw, 42px);
        font-weight: 900;
        margin: 8px 0;
    }}

    .gold {{ color: #ffcb43; }}
    .pink {{ color: #ff5799; }}

    .team-card {{
        background: rgba(4, 19, 42, .97);
        border: 1px solid var(--team-colour);
        border-radius: 14px;
        padding: 18px 8px;
        min-height: 145px;
        text-align: center;
        margin-bottom: 14px;
        box-shadow: 0 5px 18px rgba(0,0,0,.2);
    }}

    .team-emoji {{
        font-size: 30px;
    }}

    .team-short {{
        font-size: 20px;
        font-weight: 900;
        color: var(--team-colour);
        margin: 8px 0;
    }}

    .team-name-small {{
        font-size: 11px;
        color: white;
        font-weight: 600;
    }}

    .footer-text {{
        color: #9bb3d1;
        text-align: center;
        font-size: 12px;
        margin-top: 35px;
    }}
    </style>

    <div class="hero">
        <div class="eyebrow">
            CRICKET • DATA • ARTIFICIAL INTELLIGENCE
        </div>
        <h1>IPL <span>WIN PREDICTOR</span> 🏏</h1>
        <p>
            Every match has a story.
            Explore winning probabilities with machine learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================
# 5. LOAD THE MODEL AND DATA
# =========================================
MODEL_PATH = ROOT / "ipl_model.pkl"
DATA_PATH = ROOT / "clean_matches.csv"

if not MODEL_PATH.exists():
    st.error("ipl_model.pkl is missing from the project folder.")
    st.stop()

if not DATA_PATH.exists():
    st.error("clean_matches.csv is missing from the project folder.")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    matches = pd.read_csv(DATA_PATH)

except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()

if not {"team1", "team2", "city"}.issubset(matches.columns):
    st.error("The match dataset is missing required columns.")
    st.stop()

teams = sorted(
    set(matches["team1"].dropna().astype(str))
    | set(matches["team2"].dropna().astype(str))
)

cities = sorted(
    matches["city"].dropna().astype(str).unique().tolist()
)

cities = ["Unknown"] + [
    city for city in cities if city != "Unknown"
]

# =========================================
# 6. MATCH SELECTION
# =========================================
st.markdown(
    """
    <div class="panel">
        <h2>🏆 Match Arena</h2>
        <div class="muted">
            Select two teams and a venue to view the prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    team1 = st.selectbox(
        "🏏 SELECT TEAM ONE",
        teams,
        key="team_one",
    )

with right:
    opponents = [team for team in teams if team != team1]

    team2 = st.selectbox(
        "⚡ SELECT TEAM TWO",
        opponents,
        key="team_two",
    )

city = st.selectbox(
    "📍 SELECT VENUE CITY",
    cities,
    key="venue",
)

predict = st.button(
    "⚡ PREDICT WINNER",
    use_container_width=True,
)

# =========================================
# 7. PREDICTION RESULTS
# =========================================
if predict:
    try:
        input_data = pd.DataFrame([{
            "team1": team1,
            "team2": team2,
            "city": city,
        }])

        probabilities = model.predict_proba(input_data)[0]

        classes = list(
            model.named_steps["classifier"].classes_
        )

        p1 = float(probabilities[classes.index(1)])
        p2 = 1.0 - p1

        st.markdown("## ⚡ Prediction Results")

        col1, centre, col2 = st.columns([5, 1, 5])

        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="team-name">{team1}</div>
                    <div class="prob gold">{p1:.1%}</div>
                    <div class="muted">Estimated win probability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with centre:
            st.markdown(
                """
                <div style="
                    text-align:center;
                    padding-top:65px;
                    font-size:22px;
                    font-weight:900;
                    color:#62caff;">
                    VS
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="team-name">{team2}</div>
                    <div class="prob pink">{p2:.1%}</div>
                    <div class="muted">Estimated win probability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(p1)

        if p1 > p2:
            winner = team1
        elif p2 > p1:
            winner = team2
        else:
            winner = "Equal probabilities"

        st.success(f"🏆 Model favourite: {winner}")

        st.caption(
            "Estimates use historical team and venue data. "
            "They do not include live scores, wickets, toss results "
            "or current player form, and are not guarantees."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

#
# =========================================
# 8. ALL 10 IPL TEAMS WITH LOGOS
# =========================================
st.markdown("---")
st.markdown("## 🌟 Meet the IPL Teams")
st.caption("Ten teams. One trophy. Endless possibilities.")

team_logos = [
    ("CSK", "Chennai Super Kings",
     "https://scores.iplt20.com/ipl/teamlogos/CSK.png",
     "#ffcc22"),
    ("MI", "Mumbai Indians",
     "https://scores.iplt20.com/ipl/teamlogos/MI.png",
     "#2787ff"),
    ("RCB", "Royal Challengers Bengaluru",
     "https://scores.iplt20.com/ipl/teamlogos/RCB.png",
     "#ff4057"),
    ("KKR", "Kolkata Knight Riders",
     "https://scores.iplt20.com/ipl/teamlogos/KKR.png",
     "#b68aff"),
    ("SRH", "Sunrisers Hyderabad",
     "https://scores.iplt20.com/ipl/teamlogos/SRH.png",
     "#ff812d"),
    ("RR", "Rajasthan Royals",
     "https://scores.iplt20.com/ipl/teamlogos/RR.png",
     "#ff70bb"),
    ("DC", "Delhi Capitals",
     "https://scores.iplt20.com/ipl/teamlogos/DC.png",
     "#4e9cff"),
    ("PBKS", "Punjab Kings",
     "https://scores.iplt20.com/ipl/teamlogos/PBKS.png",
     "#ff5c66"),
    ("LSG", "Lucknow Super Giants",
     "https://scores.iplt20.com/ipl/teamlogos/LSG.png",
     "#46d7d7"),
    ("GT", "Gujarat Titans",
     "https://scores.iplt20.com/ipl/teamlogos/GT.png",
     "#b4c9ed"),
]

for start in range(0, len(team_logos), 5):
    columns = st.columns(5)

    for column, team in zip(
        columns, team_logos[start:start + 5]
    ):
        short, name, logo_url, colour = team

        with column:
            st.markdown(
                f"""
                <div style="
                    background:rgba(4,19,42,.97);
                    border:1px solid {colour};
                    border-radius:14px;
                    padding:14px 8px 8px;
                    text-align:center;
                    margin-bottom:12px;
                ">
                    <div style="
                        color:{colour};
                        font-size:18px;
                        font-weight:900;
                        margin-bottom:8px;
                    ">{short}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                st.image(logo_url, width=100)
            except Exception:
                st.markdown(
                    f"<h2 style='text-align:center;color:{colour}'>"
                    f"{short}</h2>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div style='text-align:center;color:white;"
                f"font-size:12px;font-weight:600;margin-bottom:18px'>"
                f"{name}</div>",
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption("🏏 Built with Python, Streamlit and Machine Learning")

# =========================================
# 9. FOOTER
# =========================================
st.markdown(
    """
    <div class="footer-text">
        BUILT WITH PYTHON • STREAMLIT • MACHINE LEARNING 🏏
    </div>
    """,
    unsafe_allow_html=True,
)
