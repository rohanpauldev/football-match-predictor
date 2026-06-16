import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Match Predictor", page_icon="⚽")
st.title("⚽ International Match Predictor")

# Load & prepare data 

@st.cache_data
def load_and_train():
    df = pd.read_csv("results.csv")

    df["goal_diff"] = df["home_score"] - df["away_score"]
    df = df.dropna(subset=["home_score", "away_score"])

    def get_result(row):
        if row["home_score"] > row["away_score"]:
            return "Home Win"
        elif row["home_score"] < row["away_score"]:
            return "Away Win"
        else:
            return "Draw"

    df["result"] = df.apply(get_result, axis=1)

    # Team stats
    home_stats = df.groupby("home_team").agg(
        home_goals_scored=("home_score", "sum"),
        home_goals_conceded=("away_score", "sum"),
        home_matches=("home_team", "count"),
    )
    away_stats = df.groupby("away_team").agg(
        away_goals_scored=("away_score", "sum"),
        away_goals_conceded=("home_score", "sum"),
        away_matches=("away_team", "count"),
    )

    team_stats = home_stats.join(away_stats, how="outer")
    team_stats["matches"] = team_stats["home_matches"] + team_stats["away_matches"]
    team_stats["goals_scored"] = team_stats["home_goals_scored"] + team_stats["away_goals_scored"]
    team_stats["goals_conceded"] = team_stats["home_goals_conceded"] + team_stats["away_goals_conceded"]
    team_stats["avg_goals_scored"] = team_stats["goals_scored"] / team_stats["matches"]
    team_stats["avg_goals_conceded"] = team_stats["goals_conceded"] / team_stats["matches"]

    home_wins = (df["home_score"] > df["away_score"]).groupby(df["home_team"]).sum()
    away_wins = (df["away_score"] > df["home_score"]).groupby(df["away_team"]).sum()
    team_stats["wins"] = home_wins.add(away_wins, fill_value=0)
    team_stats["win_rate"] = team_stats["wins"] / team_stats["matches"]

    # Recent form
    df = df.reset_index(drop=True)
    df["match_id"] = df.index

    home_form = pd.DataFrame({
        "match_id": df["match_id"],
        "team": df["home_team"],
        "date": df["date"],
    })
    home_form["points"] = np.where(
        df["home_score"] > df["away_score"], 3,
        np.where(df["home_score"] == df["away_score"], 1, 0),
    )
    home_form["side"] = "home"

    away_form = pd.DataFrame({
        "match_id": df["match_id"],
        "team": df["away_team"],
        "date": df["date"],
    })
    away_form["points"] = np.where(
        df["away_score"] > df["home_score"], 3,
        np.where(df["away_score"] == df["home_score"], 1, 0),
    )
    away_form["side"] = "away"

    form_df = pd.concat([home_form, away_form], ignore_index=True)
    form_df["date"] = pd.to_datetime(form_df["date"])
    form_df = form_df.sort_values(["team", "date"])
    form_df["recent_form_5"] = (
        form_df.groupby("team")["points"]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

    home_recent = (
        form_df[form_df["side"] == "home"][["match_id", "recent_form_5"]]
        .rename(columns={"recent_form_5": "home_recent_form_5"})
    )
    away_recent = (
        form_df[form_df["side"] == "away"][["match_id", "recent_form_5"]]
        .rename(columns={"recent_form_5": "away_recent_form_5"})
    )

    model_df = df.copy()
    model_df = model_df.merge(home_recent, on="match_id", how="left")
    model_df = model_df.merge(away_recent, on="match_id", how="left")

    global_avg_form = form_df["recent_form_5"].mean()
    model_df["home_recent_form_5"] = model_df["home_recent_form_5"].fillna(global_avg_form)
    model_df["away_recent_form_5"] = model_df["away_recent_form_5"].fillna(global_avg_form)

    model_df["home_avg_goals_scored"] = model_df["home_team"].map(team_stats["avg_goals_scored"])
    model_df["away_avg_goals_scored"] = model_df["away_team"].map(team_stats["avg_goals_scored"])
    model_df["home_avg_goals_conceded"] = model_df["home_team"].map(team_stats["avg_goals_conceded"])
    model_df["away_avg_goals_conceded"] = model_df["away_team"].map(team_stats["avg_goals_conceded"])
    model_df["home_win_rate"] = model_df["home_team"].map(team_stats["win_rate"])
    model_df["away_win_rate"] = model_df["away_team"].map(team_stats["win_rate"])

    features = [
        "home_avg_goals_scored",
        "away_avg_goals_scored",
        "home_avg_goals_conceded",
        "away_avg_goals_conceded",
        "home_win_rate",
        "away_win_rate",
        "home_recent_form_5",
        "away_recent_form_5",
        "neutral",
    ]

    model_df = model_df.dropna(subset=features)
    model_df["date"] = pd.to_datetime(model_df["date"])
    model_df = model_df.sort_values("date")

    split_index = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_index]
    test = model_df.iloc[split_index:]

    X_train = train[features]
    y_train = train["result"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Latest form per team
    latest_form = (
        form_df.groupby("team")["recent_form_5"].last().to_dict()
    )

    all_teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))

    return model, team_stats, latest_form, all_teams, features


model, team_stats, latest_form, all_teams, features = load_and_train()

# UI 

st.subheader("Select Teams")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Home Team", all_teams, index=all_teams.index("Argentina"))
with col2:
    away_team = st.selectbox("✈️ Away Team", all_teams, index=all_teams.index("Algeria"))

neutral = st.checkbox("Neutral Venue", value=True)

if home_team == away_team:
    st.warning("Please select two different teams.")
    st.stop()

missing = []
for team in [home_team, away_team]:
    if team not in team_stats.index:
        missing.append(team)

if missing:
    st.error(f"No historical data for: {', '.join(missing)}")
    st.stop()

# Prediction 

home = team_stats.loc[home_team]
away = team_stats.loc[away_team]

match = pd.DataFrame([{
    "home_avg_goals_scored": home["avg_goals_scored"],
    "away_avg_goals_scored": away["avg_goals_scored"],
    "home_avg_goals_conceded": home["avg_goals_conceded"],
    "away_avg_goals_conceded": away["avg_goals_conceded"],
    "home_win_rate": home["win_rate"],
    "away_win_rate": away["win_rate"],
    "home_recent_form_5": latest_form.get(home_team, 0),
    "away_recent_form_5": latest_form.get(away_team, 0),
    "neutral": neutral,
}])

probs = model.predict_proba(match)[0]
classes = model.classes_  # ['Away Win', 'Draw', 'Home Win']

prob_map = dict(zip(classes, probs))
home_win_pct = prob_map.get("Home Win", 0) * 100
draw_pct = prob_map.get("Draw", 0) * 100
away_win_pct = prob_map.get("Away Win", 0) * 100

# Results display 

st.divider()
st.subheader(f"📊 {home_team} vs {away_team}")

col1, col2, col3 = st.columns(3)
col1.metric(f"🏠 {home_team} Win", f"{home_win_pct:.1f}%")
col2.metric("🤝 Draw", f"{draw_pct:.1f}%")
col3.metric(f"✈️ {away_team} Win", f"{away_win_pct:.1f}%")

# Progress bar visual
st.write("")
total = home_win_pct + draw_pct + away_win_pct
st.progress(int(home_win_pct), text=f"{home_team}: {home_win_pct:.1f}%")
st.progress(int(draw_pct), text=f"Draw: {draw_pct:.1f}%")
st.progress(int(away_win_pct), text=f"{away_team}: {away_win_pct:.1f}%")

# Team stats table
st.divider()
st.subheader("📋 Team Stats")

stats_display = pd.DataFrame({
    "": ["Avg Goals Scored", "Avg Goals Conceded", "Win Rate", "Recent Form (5 games)"],
    home_team: [
        f"{home['avg_goals_scored']:.2f}",
        f"{home['avg_goals_conceded']:.2f}",
        f"{home['win_rate']*100:.1f}%",
        f"{latest_form.get(home_team, 0):.2f} pts/game",
    ],
    away_team: [
        f"{away['avg_goals_scored']:.2f}",
        f"{away['avg_goals_conceded']:.2f}",
        f"{away['win_rate']*100:.1f}%",
        f"{latest_form.get(away_team, 0):.2f} pts/game",
    ],
}).set_index("")

st.table(stats_display)
