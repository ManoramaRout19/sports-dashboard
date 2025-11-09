import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from chatbot import local_chatbot

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Sports Statistics Dashboard")

# --- LOAD CUSTOM CSS ---
def local_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

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
st.sidebar.markdown("Customize your stats view:")

sports = [s for s in sorted(df_main["Sport"].dropna().unique()) if s.lower() not in ["sport"]]
genders = [g for g in sorted(df_main["Gender"].dropna().unique()) if g.lower() not in ["gender"]]
sport = st.sidebar.radio("Sport", sports)
gender = st.sidebar.radio("Team Category", genders)

venue = st.sidebar.selectbox("Venue", ["All"] + sorted(df_main["Venue"].dropna().unique()))
team = st.sidebar.selectbox("Team (Winner)", ["All"] + sorted(df_main["Winner"].dropna().unique()))
search_name = st.sidebar.text_input("Player/Team Name")

df = pd.read_csv("matches.csv", delimiter=";")

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

st.sidebar.title("💬 SportsBot (Chat)")
user_query = st.sidebar.text_input("Your question...")
if st.sidebar.button("Send") and user_query:
    response = local_chatbot(user_query, df)
    st.sidebar.markdown(f"**Bot : Hey!** {response}")

# --- FILTER LOGIC ---
df = df_main[(df_main["Sport"] == sport) & (df_main["Gender"] == gender)]
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
if df.empty:
    st.warning("No data found for selected filters.")
    st.stop()

# --- HEADER ---
st.markdown('<h1 class="main-app-title">Sports & Statistics💡</h1>', unsafe_allow_html=True)
st.markdown(
    '<div style="display:flex;justify-content:center;">'
    '<div class="instruction-box-final">'
    'Select your filters & explore data visualizations below!'
    '</div></div>',
    unsafe_allow_html=True
)
st.markdown("---")
    
# --- KPI CARDS ---
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
st.markdown("---")

# ------------------ 📊 OVERVIEW TAB ------------------
def overview_tab(df, sport, gender):
    st.markdown('<h2 class="sport-heading">📊 Overview Dashboard</h2>', unsafe_allow_html=True)
    st.markdown('<div class="instruction-box-final">Here’s a quick glance at the current selections and overall sport statistics 💡</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

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

    # --- BAR CHART: Team Wins ---
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

    # --- TOP PLAYERS ---
    st.subheader(f"💪 Top 10 {('Run Scorers 🏏' if sport == 'Cricket' else 'Goal Scorers ⚽')}")
    top_players = df.groupby("Player")["Runs/Goals"].sum().nlargest(10).reset_index()
    if top_players.empty:
        st.warning("No player data for current filters.")
    else:
        st.dataframe(top_players, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- LINE TREND: Cumulative Wins ---
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
    st.subheader("🗺️ Venue Advantage Heatmap")
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

# ------------------ ⚽ TEAM INSIGHTS TAB ------------------
def team_insights_tab(df, sport):
    st.markdown('<h2 class="sport-heading">💥 Team Insights</h2>', unsafe_allow_html=True)
    st.markdown('<div class="instruction-box-final">Dive deep into how teams perform against others — head‑to‑head stats, consistency, and toss/venue advantages!</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TEAM HEAD‑TO‑HEAD COMPARISON ---
    st.subheader("⚔️ Head‑to‑Head Comparison")
    all_teams = sorted(list(set(df["Winner"]).union(set(df["Loser"]))))
    if len(all_teams) < 2:
        st.warning("Please select filters with at least two unique teams.")
        return

    colA, colB = st.columns(2)
    with colA:
        teamA = st.selectbox("Select Team A", all_teams, key="teamA")
    with colB:
        teamB = st.selectbox("Select Team B", all_teams, key="teamB")

    if teamA and teamB and teamA != teamB:
        h2h = df[((df["Winner"] == teamA) & (df["Loser"] == teamB)) |
                 ((df["Winner"] == teamB) & (df["Loser"] == teamA))]
        if not h2h.empty:
            summary = h2h["Winner"].value_counts().reset_index()
            summary.columns = ["Team", "Wins"]
            total_matches = len(h2h)
            fig = px.bar(summary, x="Team", y="Wins", color="Team",
                title=f"Head‑to‑Head: {teamA} vs {teamB} ({total_matches} matches)",
                template="plotly_white")
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No recorded matches between {teamA} and {teamB}.")
    else:
        st.info("Select two different teams to compare!")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # --- RADAR CHART: TEAM PERFORMANCE COMPARISON ---
    st.subheader("📊 Team Performance Radar")
    top_teams = df["Winner"].value_counts().head(5).index.tolist()
    if len(top_teams) >= 3:
        radar_df = pd.DataFrame({
            "Team": top_teams,
            "AvgScore": [df[df["Winner"] == t]["Runs/Goals"].mean() for t in top_teams],
            "Matches": [len(df[(df["Winner"] == t) | (df["Loser"] == t)]) for t in top_teams],
            "UniqueOpponents": [df[df["Winner"] == t]["Loser"].nunique() for t in top_teams],
        })
        radar_fig = px.line_polar(
            radar_df.melt(id_vars="Team"),
            r="value",
            theta="variable",
            color="Team",
            line_close=True,
            template="plotly_white",
            title=f"{sport} Top‑5 Teams Performance Radar"
        )
        radar_fig.update_traces(fill="toself", opacity=0.6)
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title_x=0.5)
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Need data for at least 3 teams to generate radar chart.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # --- TOSS IMPACT / FIRST INNINGS ADVANTAGE ---
    if sport.lower() == "cricket":
        st.subheader("🎯 Toss Decision Impact (Cricket only)")
        if "Toss Decision" in df.columns:
            toss_df = df.groupby("Toss Decision")["Winner"].count().reset_index()
            toss_df.columns = ["Decision", "Wins"]
            fig_toss = px.bar(toss_df, x="Decision", y="Wins",
                color="Decision", title="Win Counts by Toss Decision", template="plotly_white")
            fig_toss.update_layout(title_x=0.5)
            st.plotly_chart(fig_toss, use_container_width=True)
        else:
            st.info("Toss Decision column not found in this dataset.")
    else:
        st.info("Toss impact metric is only for Cricket matches.")

# ------------------ 🏅 PLAYER INSIGHTS TAB ------------------
def player_insights_tab(df, sport, gender):
    st.markdown('<h2 class="sport-heading">🏅 Player Insights</h2>', unsafe_allow_html=True)
    st.markdown('<div class="instruction-box-final">Explore individual player performance, consistency, and leaderboards!</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- PLAYER PERFORMANCE DISTRIBUTION ---
    st.subheader("🎯 Runs/Goals Distribution")
    if df.empty:
        st.warning("No data for player distribution chart.")
    else:
        fig_hist = px.histogram(df, x="Runs/Goals", nbins=20, color_discrete_sequence=["#c7b8fe"],
            title=f"{sport} ‑ Player Score/Goal Distribution")
        fig_hist.update_layout(title_x=0.5)
        st.plotly_chart(fig_hist, use_container_width=True)


    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # --- CONSISTENCY INDEX (robust final version) ---
    st.subheader("📊 Player Consistency Index")

    player_stats = df.groupby("Player")["Runs/Goals"].agg(["mean", "std", "count"]).reset_index()
    # Only show for players with at least 2 games (avoid division by zero issues)
    player_stats = player_stats[player_stats["count"] > 1].copy()
    # Replace zero or nan stds for safety; avoid divide-by-zero
    player_stats["std"].replace(0, np.nan, inplace=True)
    player_stats["Consistency Index"] = (player_stats["mean"] / player_stats["std"]).round(2)
    # Remove any nan or inf values
    player_stats = player_stats.replace([np.inf, -np.inf], np.nan).dropna(subset=["Consistency Index"])

    if player_stats.empty:
        st.warning("No players with at least two matches; please select different filters or add matches.")
    else:
        fig_consistency = px.bar(
        player_stats.sort_values(by="Consistency Index", ascending=False).head(10),
        x="Player",
        y="Consistency Index",
        color="Player",
        title=f"Top 10 Consistent Players in {sport} ({gender})",
        template="plotly_white"
    )
    fig_consistency.update_layout(title_x=0.5)
    st.plotly_chart(fig_consistency, use_container_width=True)

    # --- TOTAL CONTRIBUTION ---
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

    # --- LEADERBOARD TABLE ---
    st.subheader("🏆 Player Leaderboard")
    leaderboard = df.groupby("Player")["Runs/Goals"].sum().reset_index().sort_values("Runs/Goals", ascending=False)
    if leaderboard.empty:
        st.warning("No leaderboard data for current filters.")
    else:
        st.dataframe(leaderboard.head(15), use_container_width=True)
        st.caption("This leaderboard ranks players by total runs/goals achieved within selected filters.")

# ------------------ 🔮 PREDICTOR TAB ------------------
def predictor_tab(df):
    st.markdown('<h2 class="sport-heading">🔮 Match Predictor & Downloads</h2>', unsafe_allow_html=True)
    st.markdown('<div class="instruction-box-final">Predict potential match outcomes and download your data exports!</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- DATA DOWNLOAD BUTTON ---
    st.subheader("💾 Export Filtered Dataset")
    st.markdown("---")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name='filtered_matches.csv',
        mime='text/csv'
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # --- SIMPLE PREDICTION SECTION ---
    st.subheader("⚡ Predict Possible Match Winner")

    # Basic validation
    if df["Winner"].nunique() < 2:
        st.warning("Need at least two unique teams to simulate prediction.")
        return

    teams = sorted(df["Winner"].unique())
    team_a = st.selectbox("Select Team A", teams, key="pred_teamA")
    team_b = st.selectbox("Select Team B", [t for t in teams if t != team_a], key="pred_teamB")

    st.markdown("<br>", unsafe_allow_html=True)

    # Prepare a simple encoding model for demo purposes
    le = LabelEncoder()
    df_enc = df.copy()
    df_enc["Winner_enc"] = le.fit_transform(df_enc["Winner"])
    df_enc["Loser_enc"] = le.fit_transform(df_enc["Loser"])
    df_enc = df_enc.dropna(subset=["Runs/Goals"])

    # Create binary target if same Sport context exists
    X = df_enc[["Winner_enc", "Loser_enc", "Runs/Goals"]]
    y = (df_enc["Winner_enc"] > df_enc["Loser_enc"]).astype(int)

    if len(X) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        accuracy = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)

        teamA_enc = le.transform([team_a])[0] if team_a in le.classes_ else 0
        teamB_enc = le.transform([team_b])[0] if team_b in le.classes_ else 0
        sample_input = np.array([[teamA_enc, teamB_enc, df_enc["Runs/Goals"].mean()]])
        prediction = model.predict(sample_input)[0]

        predicted_winner = team_a if prediction == 1 else team_b
        st.success(f"🏆 Predicted Winner: **{predicted_winner}**")
        st.caption(f"Model Accuracy: {accuracy}% (using demo logistic regression)")
    else:
        st.warning("Not enough data for reliable prediction. Please apply broader filters.")
        
# --- LOAD UPCOMING MATCHES DATA ---
@st.cache_data
def load_upcoming():
    df_up = pd.read_csv("upcoming_matches.csv", sep=';')
    df_up.columns = df_up.columns.str.strip()
    df_up['Date'] = pd.to_datetime(df_up['Date'], errors='coerce').dt.date
    return df_up

try:
    df_upcoming = load_upcoming()
except Exception:
    df_upcoming = pd.DataFrame()

# --- UPCOMING MATCHES SECTION ---
st.markdown('## 🕒 Upcoming Matches', unsafe_allow_html=True)
today = pd.Timestamp.now().date()
upcoming_mask = (df_upcoming['Sport'] == sport) & (df_upcoming['Gender'] == gender) & (df_upcoming['Date'] > today)
upcoming = df_upcoming[upcoming_mask]
if not upcoming.empty:
    for _, row in upcoming.iterrows():
        st.markdown(f"""**{row['TeamA']} vs {row['TeamB']}**  
        📍 Venue: {row['Venue']}  
        📅 Date: {row['Date'].strftime('%d %b %Y')}  
        <br>""", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
else:
    st.info("No upcoming matches for the selected category.")


# ------------------ DASHBOARD TABS ------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Team Insights", "Player Insights", "Predictor"])
with tab1:
    overview_tab(df, sport, gender)
with tab2:
    team_insights_tab(df, sport)
with tab3:
    player_insights_tab(df, sport, gender)
with tab4:
    predictor_tab(df)

st.markdown("---")
st.markdown('### Predict Upcoming Match Outcome!', unsafe_allow_html=True)

if not upcoming.empty:
    options = upcoming.apply(lambda x: f"{x['TeamA']} vs {x['TeamB']} at {x['Venue']} ({x['Date']})", axis=1)
    selected = st.selectbox(
        "Select an upcoming match",
        options,
        key="predict_upcoming_select"
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
        st.success(f"🏆 Predicted Winner: **{predicted}** (Win %: {win_pct} based on {total_matches} previous matches)")
    else:
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
                st.success(f"🏆 Predicted Winner: **{team_a}** (Based on available stats)")
            elif score_b > score_a:
                st.success(f"🏆 Predicted Winner: **{team_b}** (Based on available stats)")
            else:
                st.info("Teams are evenly matched based on available stats.")
        elif not team_a_df.empty:
            st.success(f"🏆 Predicted Winner: **{team_a}** (Only stats available for {team_a})")
        elif not team_b_df.empty:
            st.success(f"🏆 Predicted Winner: **{team_b}** (Only stats available for {team_b})")
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
                Manorama Rout
            </a>
        </p> 
    </footer>
""", unsafe_allow_html=True)

# streamlit run C:\Users\manor\Desktop\data-viz-project.py\streamlit_app.py