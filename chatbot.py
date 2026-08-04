import pandas as pd
from datetime import datetime

def local_chatbot(user_query, df):
    query = user_query.strip().lower()
    if "next match" in query or "upcoming match" in query:
        
        return "Sorry, I currently focus on historical statistics. Please try questions about wins or top players."

    # --- 2. WHO WON THE MOST (Top Team) ---
    if "most wins" in query or "best team" in query or "top team" in query:
        if df.empty:
            return "The current filter selection has no data to analyze."
        
        # Calculate the team with the most wins in the current filtered view
        winner_counts = df['Winner'].value_counts()
        if not winner_counts.empty:
            top_team = winner_counts.index[0]
            wins = winner_counts.iloc[0]
            return f"The team with the most wins in the currently selected matches ({df['Sport'].iloc[0]}) is **{top_team}** with {wins} victories."
        else:
            return "I couldn't identify the winner counts in the current selection."

    # --- 3. TOP PLAYER (Total Runs/Goals) ---
    if "top player" in query or "most goals" in query or "most runs" in query:
        if 'Player' in df.columns and 'Runs/Goals' in df.columns and not df.empty:
            
            player_stats = df.groupby('Player')['Runs/Goals'].sum().nlargest(1)
            if not player_stats.empty:
                top_player = player_stats.index[0]
                total_score = player_stats.iloc[0]
                metric = "runs" if df['Sport'].iloc[0].lower() == 'cricket' else "goals"
                return f"The top performing player is **{top_player}** with a total of {total_score} {metric} in the filtered matches."
            else:
                return "No player contribution data found in the current selection."
        else:
            return "Player or Runs/Goals column is missing from the data for this analysis."

    # --- 4. AVERAGE SCORE ---
    if "average score" in query or "avg runs" in query or "avg goals" in query:
        if 'Runs/Goals' in df.columns and not df.empty:
            avg_score = df['Runs/Goals'].mean()
            metric = "runs" if df['Sport'].iloc[0].lower() == 'cricket' else "goals"
            return f"The average score (Runs/Goals) for the filtered matches is **{avg_score:.2f}** {metric} per match."
        else:
            return "Runs/Goals column is missing or data is empty for average score calculation."

    # --- 5. SPORT-SPECIFIC WINNERS LIST (Simplified from your original) ---

    if "winners list" in query or "all winners" in query:
        current_sport = df['Sport'].iloc[0]
        winners = df['Winner'].unique()
        return f"Winners in the filtered {current_sport} category are: " + ", ".join(winners)
    
    # --- DEFAULT RESPONSE ---
    else:
        return (
            "Sorry, I didn't understand that. I can currently answer questions like: "
            "**'who won the most'**, **'top player'**, **'average score'**, or **'winners list'**."
        )