import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import numpy as np
from chatbot import local_chatbot

# --- DARK MODE INITIALIZATION ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
    
# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Sports Statistics Dashboard")

# --- LOAD CUSTOM CSS ---
def local_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            css_code = f.read()
            # Apply the CSS code
            st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)
    except Exception:
        pass

# Function to apply dark mode class to the entire app body
def apply_theme_class():
    if st.session_state.dark_mode:
        theme_class = "dark-mode-theme"
    else:
        theme_class = ""
    
    # This injects the class directly into the highest-level Streamlit container
    st.markdown(f'<div class="{theme_class}">', unsafe_allow_html=True) 
    
local_css("style.css")

# --- DATA LOAD ---
@st.cache_data
def load_data():
    df = pd.read_csv("matches.csv", sep=';')
    df.columns = df.columns.str.strip()

    # Ensure proper numeric + date types
    df['Runs/Goals'] = pd.to_numeric(df['Runs/Goals'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

    return df

REQUIRED_COLS = ['Winner', 'Loser', 'Player', 'Runs/Goals', 'Venue', 'Date', 'Sport', 'Gender']

try:
    df_main = load_data()
    if df_main.empty or not all(col in df_main.columns for col in REQUIRED_COLS):
        st.error("🚫 Data file missing required columns.")
        st.stop()
except Exception:
    st.error("File Not Found or cannot load data!")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.title("✨ Dashboard Filters")
# --- DARK MODE TOGGLE ---

js_code = f"""
<script>
    const container = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
    if (container) {{
        if ({'true' if st.session_state.dark_mode else 'false'}) {{
            container.classList.add('dark-mode-theme');
        }} else {{
            container.classList.remove('dark-mode-theme');
        }}
    }}
</script>
"""

# The button handles the state change and triggers a rerun
if st.sidebar.button(
    ("💡" if st.session_state.dark_mode else "🌙"),
    key="js_mode_toggle"
):
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()
    
st.components.v1.html(js_code, height=0, width=0)

st.sidebar.markdown("Customize your stats view:")

sports = [s for s in sorted(df_main["Sport"].dropna().unique()) if s.lower() not in ["sport"]]
genders = [g for g in sorted(df_main["Gender"].dropna().unique()) if g.lower() not in ["gender"]]
sport = st.sidebar.radio("Sport", sports)
gender = st.sidebar.radio("Team Category", genders)
st.sidebar.title("💬 SportsBot (Chat)")
user_query = st.sidebar.text_input("Your question...")
if st.sidebar.button("Send") and user_query:
    response = local_chatbot(user_query, df_main) 
    st.sidebar.markdown(f"*Hey!* {response}")
search_name = st.sidebar.text_input("Player/Team Name")
venue = st.sidebar.selectbox("Venue", ["All"] + sorted(df_main["Venue"].dropna().unique()))
team = st.sidebar.selectbox("Team (Winner)", ["All"] + sorted(df_main["Winner"].dropna().unique()))

st.markdown("""
<style>
/* Mimics floating look for sidebar */
[data-testid='stSidebar'] {
    background: linear-gradient(135deg, #f4e6ff 0%, #d4eaff 100%);
    box-shadow: 0 2px 14px #7046ff55;
    border-radius: 15px 0 0 15px;
}
.stTextInput, .stButton, .stMarkdown {
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

# --- FILTER LOGIC ---
df = df_main[(df_main["Sport"] == sport) & (df_main["Gender"] == gender)].copy()
if venue != "All":
    df = df[df["Venue"] == venue]
if team != "All":
    df = df[df["Winner"] == team]
if search_name:
    search = search_name.lower()
    df = df[
        df["Player"].str.lower().str.contains(search, na=False) |
        df["Winner"].str.lower().str.contains(search, na=False) |
        df["Loser"].str.lower().str.contains(search, na=False)
    ]
# NOTE: Removed the df.empty check with st.stop() here as it prevents the app from running if filters are too strict. The tabs handle the empty state now.
    
# HEADER
st.markdown('<h1 class="main-app-title">Sports & Statistics💡</h1>', unsafe_allow_html=True)
st.markdown(
    '<div style="display:flex;justify-content:center;">'
    '<div class="instruction-box-final">'
    'Select your filters & explore data visualizations below!'
    '</div></div>',
    unsafe_allow_html=True
)
    
# KPI CARDS
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Matches", value=len(df))
with col2:
    unique_teams = df["Winner"].nunique()
    st.metric(label="Unique Teams", value=unique_teams)
with col3:
    avg_score = round(df["Runs/Goals"].mean(), 2)
    st.metric(label=f"Avg Score ({'Runs' if sport=='Cricket' else 'Goals'})", value=avg_score)

st.markdown("<br>", unsafe_allow_html=True)


# Helper function for safe percentage calculation (Defined here to avoid global/scope issues)
def calculate_win_pct(wins, total):
    """Calculates win percentage safely, handling division by zero."""
    if total == 0:
        return 0.0
    return round(wins / total * 100, 1)

#  📊 OVERVIEW TAB 
def overview_tab(df, sport, gender):
    st.markdown('<h2 class="sport-heading">📊 Overview Dashboard</h2>', unsafe_allow_html=True)
    st.markdown('<div class="instruction-box-final">Here’s a quick glance at the current selections and overall sport statistics 💡</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available for the current filters.")
        return

    # --- PIE CHART: Win Share by Teams ---
    st.subheader("🏆 Win Share by Teams")
    win_counts = df["Winner"].value_counts().reset_index()
    win_counts.columns = ["Team", "Wins"]
    if win_counts.empty:
        st.warning("No team win data available for current filters.")
    else:
        pie_fig = px.pie(win_counts, names="Team", values="Wins",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title=f"{sport} Team Win Distribution")
        pie_fig.update_layout(title_x=0.5)
        st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # BAR CHART: Team Wins
    st.subheader("📈 Team Performance (Total Wins)")
    team_wins = df["Winner"].value_counts().reset_index()
    team_wins.columns = ["Team", "Wins"]
    if team_wins.empty:
        st.warning("No team performance data available for current filters.")
    else:
        fig_bar = px.bar(team_wins, x="Team", y="Wins", color="Team",
                 title=f"{sport} Total Wins Per Team", template="plotly_white")
        fig_bar.update_layout(title_x=0.5)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TOP PLAYERS
    st.subheader(f"💪 Top 10 {('Run Scorers 🏏' if sport == 'Cricket' else 'Goal Scorers ⚽')}")
    top_players = df.groupby("Player")["Runs/Goals"].sum().nlargest(10).reset_index()
    if top_players.empty:
        st.warning("No player data for current filters.")
    else:
        st.dataframe(top_players, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # LINE TREND: Cumulative Wins
    st.subheader("📅 Cumulative Wins Over Time")
    win_trend = df.groupby(["Date", "Winner"]).size().reset_index(name="Wins")
    if win_trend.empty:
        st.warning("Not enough match data for win timeline.")
    else:
        win_trend["Cumulative Wins"] = win_trend.groupby("Winner")["Wins"].cumsum()
        fig_line = px.line(win_trend, x="Date", y="Cumulative Wins", color="Winner",
            title=f"{sport} Cumulative Wins Timeline", template="plotly_white")
        fig_line.update_layout(title_x=0.5)
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- HEATMAP: Venue Advantage ---
    st.subheader("🗺 Venue Advantage Heatmap")
    heatmap_data = df.groupby(["Venue", "Winner"]).size().unstack(fill_value=0)
    if heatmap_data.empty or heatmap_data.shape[1] == 0:
        st.warning("No venue data for heatmap.")
    else:
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#f8f6ff')
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                    cbar_kws={"label": "Wins"})
        ax.set_title(f"{sport} Team Wins by Venue (Filtered)", fontsize=14, pad=12)
        ax.set_xlabel("Winning Team 🏆")
        ax.set_ylabel("Venue 📍")
        st.pyplot(fig)
        st.caption("Darker colors show dominance — the 'home ground advantage' indicator.")

# ⚽ TEAM INSIGHTS TAB 
def team_insights_tab(df, sport):
    st.markdown('<h2 class="sport-heading">💥 Team Insights</h2>', unsafe_allow_html=True)
    st.markdown('<div class="instruction-box-final">Dive deep into how teams perform against others — head‑to‑head stats, consistency, and toss/venue advantages!</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # TEAM HEAD‑TO‑HEAD COMPARISON (Global Fallback for Selection)
    st.subheader("⚔ Head‑to‑Head Comparison")
    
    # Use global data for selection options if filtered data is too sparse
    all_teams_global = sorted(list(set(df_main["Winner"]).union(set(df_main["Loser"]))))
    
    if len(all_teams_global) < 2:
        st.warning("Not enough teams in the entire dataset for comparison.")
        return

    colA, colB = st.columns(2)
    with colA:
        teamA = st.selectbox("Select Team A", all_teams_global, key="teamA_insight") 
    with colB:
        teamB = st.selectbox("Select Team B", all_teams_global, key="teamB_insight")

    if teamA and teamB and teamA != teamB:
        # ALWAYS search the full main data for H2H records between the two selected teams
        h2h = df_main[((df_main["Winner"] == teamA) & (df_main["Loser"] == teamB)) |
                     ((df_main["Winner"] == teamB) & (df_main["Loser"] == teamA))]
        
        if not h2h.empty:
            summary = h2h["Winner"].value_counts().reset_index()
            summary.columns = ["Team", "Wins"]
            total_matches = len(h2h)
            
            # Robust Win Count Extraction
            win_a = summary[summary['Team'] == teamA]['Wins'].sum()
            win_b = summary[summary['Team'] == teamB]['Wins'].sum()
            
            # Calculate percentages
            pct_a = calculate_win_pct(win_a, total_matches)
            pct_b = calculate_win_pct(win_b, total_matches)
            
            # Display key metrics
            col_metric_a, col_metric_b = st.columns(2)
            with col_metric_a:
                st.metric(label=f"Wins for {teamA} (out of {total_matches})", 
                          value=win_a, 
                          delta=f"{pct_a}%")
            with col_metric_b:
                st.metric(label=f"Wins for {teamB} (out of {total_matches})", 
                          value=win_b, 
                          delta=f"{pct_b}%")
            
            # Create and display the chart
            fig = px.bar(summary, x="Team", y="Wins", color="Team",
                title=f"Head‑to‑Head: {teamA} vs {teamB} (Total Matches: {total_matches})",
                template="plotly_white")
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info(f"No recorded matches found between {teamA} and {teamB} in the dataset.")
    else:
        st.info("Select two different teams to compare!")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # TOSS IMPACT / FIRST INNINGS ADVANTAGE (Global Fallback)
    if sport.lower() == "cricket":
        st.subheader("🎯 Toss Decision Impact (Cricket only)")
        
        # Use df_main filtered only by Sport='Cricket' for robust Toss data
        data_source_toss = df_main[df_main["Sport"].str.strip() == "Cricket"].copy()
        
        if "Toss Decision" not in data_source_toss.columns:
            st.info("Toss Decision column not found in the dataset.")
        else:
            toss_df = data_source_toss.groupby("Toss Decision")["Winner"].count().reset_index()
            toss_df.columns = ["Decision", "Wins"]
            
            if toss_df.empty or toss_df['Wins'].sum() == 0:
                 st.info("No recorded Toss Decision data in the dataset.")
            else:
                fig_toss = px.bar(toss_df, x="Decision", y = "Wins",
                 color="Decision", 
                 title="Win Counts by Toss Decision (All Cricket Data)", 
                 template="plotly_white",
                 color_discrete_sequence=['#9C27B0', '#00BCD4'])
                fig_toss.update_layout(title_x=0.5)
                st.plotly_chart(fig_toss, use_container_width=True)
    else:
        st.info("Toss impact metric is only for Cricket matches.")

# 🏅 PLAYER INSIGHTS TAB 
def player_insights_tab(df, sport, gender):
    st.markdown('<h2 class="sport-heading">🏅 Player Insights</h2>', unsafe_allow_html=True)
    st.markdown('<div class="instruction-box-final">Explore individual player performance, consistency, and leaderboards!</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # PLAYER PERFORMANCE DISTRIBUTION 
    st.subheader("🎯 Runs/Goals Distribution")
    if df.empty:
        st.warning(f"No match data available for {sport} ({gender}) after applying filters.")
        return 
    else:
        fig_hist = px.histogram(df, x="Runs/Goals", nbins=20, color_discrete_sequence=["#c7b8fe"],
            title=f"{sport} ‑ Player Score/Goal Distribution")
        fig_hist.update_layout(title_x=0.5)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
   
    # TOTAL CONTRIBUTION 
    st.subheader("⚡ Total Contribution to Team Wins")
    contrib_df = df.groupby(["Winner", "Player"])["Runs/Goals"].sum().reset_index()
    contrib_df = contrib_df.groupby("Player")["Runs/Goals"].sum().nlargest(10).reset_index()
    if contrib_df.empty:
        st.warning("No contribution data for current filters.")
    else:
        fig_contrib = px.bar(contrib_df, x="Player", y="Runs/Goals", color="Player",
            title="Top 10 Impact Players", template="plotly_white")
        fig_contrib.update_layout(title_x=0.5)
        st.plotly_chart(fig_contrib, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

   # PLAYER PERFORMANCE OVER TIME 
    st.subheader("📆 Performance Timeline")
    trend_df = df.groupby(["Date", "Player"])["Runs/Goals"].sum().reset_index()
    if trend_df.empty:
        st.warning("No matches found for timeline.")
    else:
        default_players = trend_df["Player"].value_counts().nlargest(2).index.tolist()
        selected_players = st.multiselect(
            "Select Players to Compare",
            options=df["Player"].unique(),
            default=default_players
        )
        filtered = trend_df[trend_df["Player"].isin(selected_players)]
        if filtered.empty:
            st.info("No player data found for chosen filters.")
        else:
            line_fig = px.line(
                filtered,
                x="Date",
                y="Runs/Goals",
                color="Player",
                title="Player Performance Over Time (Trend)",
                template="plotly_white"
            )
            line_fig.update_traces(mode="markers+lines")
            line_fig.update_layout(title_x=0.5)
            st.plotly_chart(line_fig, use_container_width=True)

# LEADERBOARD TABLE 
st.subheader("🏆 Player Leaderboard")

leaderboard = df.groupby("Player").agg(
    Total_Score=('Runs/Goals', 'sum'),
    Total_Matches=('Date', 'count'),
    Average_Score=('Runs/Goals', 'mean') 
).reset_index().sort_values("Total_Score", ascending=False)

leaderboard['Average_Score'] = leaderboard['Average_Score'].round(2)

if sport.lower() == 'cricket':
    leaderboard.rename(columns={'Average_Score': 'Runs/Match (Efficiency Index)'}, inplace=True)
elif sport.lower() == 'football':
    leaderboard.rename(columns={'Average_Score': 'Goals/Match'}, inplace=True)

if leaderboard.empty:
    st.warning("No leaderboard data for current filters.")
else:
    score_col = 'Total Runs' if sport.lower() == 'cricket' else 'Total Goals'
    leaderboard.rename(columns={'Total_Score': score_col}, inplace=True)
    
    # Select final columns to display
    leaderboard = leaderboard[['Player', score_col, 'Total_Matches', leaderboard.columns[-1]]]
    
    st.dataframe(leaderboard.head(15), use_container_width=True, hide_index=True)
    st.caption("This leaderboard ranks players by total contribution (runs/goals) achieved within selected filters.")
    
# LOAD UPCOMING MATCHES DATA

@st.cache_data
def load_upcoming():
    df_up = pd.read_csv("upcoming_matches.csv", sep=';')
    df_up.columns = ['TeamA', 'TeamB', 'Venue', 'Date', 'Sport', 'Gender']
    df_up['Date'] = pd.to_datetime(df_up['Date'], errors='coerce').dt.date
    return df_up

try:
    df_upcoming = load_upcoming()
except Exception:
    df_upcoming = pd.DataFrame()

# DASHBOARD TABS (Now 3 Tabs: Overview, Team, Player) 
tab1, tab2, tab3 = st.tabs(["Overview", "Team Insights", "Player Insights"]) # Added Team Insights tab
with tab1:
    overview_tab(df, sport, gender)
with tab2: # Team Insights is now tab 2
    team_insights_tab(df, sport)
with tab3: # Player Insights is now tab 3
    player_insights_tab(df, sport, gender)

# START: UPCOMING MATCHES & PREDICTION (SIDE-BY-SIDE)

st.markdown("---")
col_match_list, col_match_predict = st.columns(2)

with col_match_list:
    st.markdown('## 🕒 Upcoming Matches', unsafe_allow_html=True)
    today = pd.Timestamp.now().date()
    df_upcoming_cleaned = df_upcoming.copy()
    df_upcoming_cleaned['Sport'] = df_upcoming_cleaned['Sport'].str.strip()
    df_upcoming_cleaned['Gender'] = df_upcoming_cleaned['Gender'].str.strip()

    upcoming_mask = (df_upcoming_cleaned['Sport'] == sport) & \
                    (df_upcoming_cleaned['Gender'] == gender) & \
                    (df_upcoming_cleaned['Date'] > today)

    upcoming = df_upcoming_cleaned[upcoming_mask]
    
    # Enhanced code block in main.py 
    if not upcoming.empty:
        st.markdown("""
            <style>
            .match-list-item {
                background: linear-gradient(90deg, #f0f7ff 5%, #fff 98%);
                border: 1px solid #e0daff;
                padding: 15px 25px;
                margin: 10px 0;
                border-radius: 10px;
                font-size: 1em;
                color: var(--deep);
                box-shadow: 0 1px 10px #e4daff50;
            }
            .match-list-title {
                font-weight: 600;
                color: #9a7efd; /* Use the accent color */
                margin-right: 15px;
            }
            .match-list-detail {
                font-size: 0.9em;
                color: #5d5d88;
            }
            </style>
        """, unsafe_allow_html=True)
        
        for _, row in upcoming.iterrows():
            # Create a single, responsive list item
            st.markdown(f"""
            <div class="match-list-item">
                <span class="match-list-title">{row['TeamA']} vs {row['TeamB']}</span>
                <span class="match-list-detail">📍 {row['Venue']}</span>
                <span class="match-list-detail">📅 {row['Date'].strftime('%d %b %Y')}</span>
                <span class="match-list-detail" style="float: right;">[{row['Gender']} {row['Sport']}]</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("No upcoming matches for the selected category.")


with col_match_predict:
    st.markdown('## Predict Upcoming Match Outcome!', unsafe_allow_html=True)
    st.markdown("---")

    if not upcoming.empty:
        options = upcoming.apply(lambda x: f"{x['TeamA']} vs {x['TeamB']} at {x['Venue']} ({x['Date']})", axis=1)
        selected = st.selectbox(
            "Select an upcoming match",
            options,
            key="predict_upcoming_select_bottom" # Changed key to avoid conflict
        )
        
        match_row = upcoming.loc[
            options == selected
        ].iloc[0]
        team_a, team_b = match_row['TeamA'], match_row['TeamB']

        # Universal predictor for upcoming matches
        h2h = df_main[((df_main["Winner"] == team_a) & (df_main["Loser"] == team_b)) |
                      ((df_main["Winner"] == team_b) & (df_main["Loser"] == team_a))]

        if not h2h.empty and h2h["Winner"].nunique() >= 2:
            win_counts = h2h["Winner"].value_counts()
            total_matches = len(h2h)
            predicted = win_counts.idxmax()
            win_pct = round(100 * win_counts.max() / total_matches, 2)
            st.success(f"🏆 Predicted Winner: *{predicted}* (Win %: {win_pct} based on {total_matches} previous matches)")
        else:
            # Fallback prediction based on overall win rates
            team_a_df = df_main[(df_main["Winner"] == team_a) | (df_main["Loser"] == team_a)]
            team_b_df = df_main[(df_main["Winner"] == team_b) | (df_main["Loser"] == team_b)]
            
            if not team_a_df.empty and not team_b_df.empty:
                win_a = team_a_df["Winner"].value_counts().get(team_a, 0)
                win_b = team_b_df["Winner"].value_counts().get(team_b, 0)
                win_pct_a = win_a / len(team_a_df) if len(team_a_df) > 0 else 0
                win_pct_b = win_b / len(team_b_df) if len(team_b_df) > 0 else 0
                avg_a = team_a_df["Runs/Goals"].mean() if "Runs/Goals" in team_a_df else 0
                avg_b = team_b_df["Runs/Goals"].mean() if "Runs/Goals" in team_b_df else 0
                score_a = win_pct_a * 0.6 + (avg_a / (avg_a + avg_b)) * 0.4 if (avg_a + avg_b) > 0 else win_pct_a
                score_b = win_pct_b * 0.6 + (avg_b / (avg_a + avg_b)) * 0.4 if (avg_a + avg_b) > 0 else win_pct_b
                
                if score_a > score_b:
                    st.success(f"🏆 Predicted Winner: *{team_a}* (Based on available stats)")
                elif score_b > score_a:
                    st.success(f"🏆 Predicted Winner: *{team_b}* (Based on available stats)")
                else:
                    st.info("Teams are evenly matched based on available stats.")
            elif not team_a_df.empty:
                st.success(f"🏆 Predicted Winner: *{team_a}* (Only stats available for {team_a})")
            elif not team_b_df.empty:
                st.success(f"🏆 Predicted Winner: *{team_b}* (Only stats available for {team_b})")
            else:
                st.warning("No stats available for either team. Prediction is random.")

    else:
        st.warning("No upcoming matches available for prediction.")

# --- Footer ---
st.markdown("---")
st.markdown("""
    <footer style="
        background: linear-gradient(90deg, #e4f0fb 0%, #ffe2f6 100%);
        color: #3a3670;
        text-align: center;
        padding: 22px 0 18px 0;
        font-size: 1.05em;
        border-top: 2px solid #c7b8fe;
        border-radius: 0 0 12px 12px;
        margin-top: 50px;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 -4px 20px #e0dafc70;
    ">
        <p style="margin: 0;">
            © 2025 <span style="color:#9a7efd; font-weight:600;">Sports Statistics</span> — Built by 
            <a href="#" target="_blank" style="color:#9a7efd; text-decoration:none; font-weight:600;">
                MANORAMA ROUT
            </a>
        </p> 
    </footer>
""", unsafe_allow_html=True)

# streamlit run C:\Users\manor\Desktop\data-viz-project.py\streamlit_app.py
# streamlit run streamlit_app.py