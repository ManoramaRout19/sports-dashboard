# chatbot.py

import pandas as pd
from datetime import datetime

def local_chatbot(user_query, df):
    query = user_query.strip().lower()
    now = datetime.now().strftime("%Y-%m-%d")
    if "next football match" in query:
        matches = df[(df['Sport'].str.lower() == 'football') & (df['Date'] >= now)].sort_values(by='Date')
        if not matches.empty:
            m = matches.iloc[0]
            return f"Next football match: {m['Winner']} vs {m['Loser']} at {m['Venue']} on {m['Date']}."
        else:
            return "No upcoming football matches found."
    elif "cricket winners" in query:
        winners = df[df['Sport'].str.lower() == 'cricket']['Winner'].unique()
        return "Cricket winners: " + ", ".join(winners)
    else:
        return "Sorry, I can answer questions like 'next football match' or 'cricket winners'."
