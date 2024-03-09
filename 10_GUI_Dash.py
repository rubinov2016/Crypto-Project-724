# Import necessary libraries
import dash
from dash import dash_table
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from correlation import get_top_correlated_stocks
import plotly.figure_factory as ff

import seaborn as sns
import matplotlib.pyplot as plt
# Load data
df = pd.read_csv('crypto_data_clean.csv')

df_reduced = pd.read_csv('crypto_data_reduced.csv')
df_reduced = df_reduced[['Index', 'Cluster']]

df_reduced.rename(columns={'Index': 'Stock'}, inplace=True)
df_reduced['Cluster'] = pd.to_numeric(df_reduced['Cluster'], errors='coerce')

# # Load your data into a pandas DataFrame
# df = pd.read_csv('crypto_data_reduced.csv')
# df = df[['Index', 'Cluster']]  # Assuming these are the columns you want to keep

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout

app.layout = html.Div([
    # Create a sidebar for navigation
    html.Div([
        html.H2("Navigation", className="display-4"),
        html.Hr(),
        html.P("Cryptocurrencies analysis", className="lead"),
        dcc.Link("Home", href="/", className="nav-link"),
        dcc.Link("Refresh data", href="/loading", className="nav-link"),
        dcc.Link("Feature reduction", href="/pca", className="nav-link"),
        dcc.Link("Clustering", href="/clustering", className="nav-link"),
        dcc.Link("EDA", href="/eda", className="nav-link"),
        dcc.Link("Model Training", href="/models", className="nav-link"),
        dcc.Link("Forecasting", href="/forecasting", className="nav-link"),
    ], className="sidebar"),

    # Create a content div to display page content
    html.Div(id="page-content", className="content"),

    # Add a URL bar but hide it from view
    dcc.Location(id="url", refresh=False),
])
# Define callback to update page content based on URL
@app.callback(Output("page-content", "children"),
              [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/clustering":
        # Here we add a DataTable to the "Clustering" page
        return html.Div([
            html.H1("Clustering Page Content"),

            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i, 'deletable': True, "type": "numeric" if i == "Cluster" else "text"} for i in df_reduced.columns],
                data=df_reduced.to_dict('records'),
                filter_action="native",  # Enable filtering
                sort_action="native",  # Enable sorting
                editable=True,  # Allow cell editing
                style_table={'width': '50%'},  # Set the width of the table
            ),
        ])
    elif pathname == "/eda":
        html.H1(children='Stock Correlation Analysis'),
        # List of stocks to choose from
        stock_options = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD']
        # Specify the target stock symbol you're interested in
        target_symbol = 'BTC-USD'  # Replace with your target stock symbol

        # Use the function to get top correlated stocks
        top_positive, top_negative = get_top_correlated_stocks(target_symbol, df, n=10)
        # Sort the negative correlations so the most negative come first
        top_negative = top_negative.sort_values()

        top_positive_df = top_positive.reset_index()
        top_positive_df.columns = ['Stock', 'Correlation']
        top_negative_df = top_negative.reset_index()
        top_negative_df.columns = ['Stock', 'Correlation']

        z_positive = [top_positive_df['Correlation'].values.tolist()]  # Wrapping in another list to make it 2D
        x_positive = top_positive_df['Stock'].values.tolist()  # Ensuring 'x' is a list

        # Similar for top_negative_df
        z_negative = [top_negative_df['Correlation'].values.tolist()]  # 2D
        x_negative = top_negative_df['Stock'].values.tolist()  # List

        # Generate heatmaps using Plotly Figure Factory
        # Now use these in your heatmap creation
        # Define a custom colorscale
        colorscale = [
            # Assign deep red to the most negative correlations
            [0.0, 'rgb(178,24,43)'],

            # Transition to lighter red for less negative correlations
            [0.25, 'rgb(239,138,98)'],

            # Use white for zero correlation
            [0.5, 'rgb(255,255,255)'],

            # Transition to lighter green for positive correlations
            [0.75, 'rgb(103,169,207)'],

            # Assign deep green to the most positive correlations
            [1.0, 'rgb(26,150,65)']
        ]
        fig_positive = ff.create_annotated_heatmap(
            z=z_positive,
            x=x_positive,
            annotation_text=[top_positive_df['Correlation'].round(2).astype(str).values.tolist()],
            # Ensuring this is also correctly structured
            colorscale='Greens',
            showscale=True
        )

        fig_positive.update_layout(title='Top 10 Positive Correlations', xaxis={'title': 'Stock'},
                                   yaxis={'title': 'Correlation'})

        annotation_text_negative = [[f"{val:.2f}" for val in z_negative[0]]]  # Formatted text annotations

        # Now use these in your heatmap creation
        fig_negative = ff.create_annotated_heatmap(
            z=z_negative,
            x=x_negative,
            annotation_text=annotation_text_negative,
            # Ensuring this is also correctly structured
            colorscale=colorscale,
            showscale=True
        )

        fig_negative.update_layout(title='Top 10 Negative Correlations', xaxis={'title': 'Stock'},
                                   yaxis={'title': 'Correlation'})
        # Return a Div containing the heatmap Graph for the EDA page
        return html.Div([
            html.H1("EDA Page Content"),
            dcc.Graph(id='positive-correlations', figure=fig_positive),
            dcc.Graph(id='negative-correlations', figure=fig_negative)

        ])
        return html.H1("EDA Page Content")
    elif pathname == "/models":
        return html.H1("Models Page Content")
    elif pathname == "/forecasting":
        return html.H1("Forecasting Page Content")
    else:
        # Default to home when nothing else is matched
        return html.H1("Home Page Content")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)