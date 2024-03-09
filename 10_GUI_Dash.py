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
import plotly.graph_objs as go
from scipy.stats import skew, kurtosis

import seaborn as sns
import matplotlib.pyplot as plt
# Load data
df = pd.read_csv('crypto_data_clean.csv')
df_stocks = df.copy()
if df_stocks.index.name is None or df_stocks.index.name != df_stocks.columns[0]:
    df_stocks = df_stocks.set_index(df.columns[0])
df_stocks = df_stocks.transpose()

df_reduced = pd.read_csv('crypto_data_reduced.csv')
df_reduced = df_reduced[['Index', 'Cluster']]

df_reduced.rename(columns={'Index': 'Stock'}, inplace=True)
df_reduced['Cluster'] = pd.to_numeric(df_reduced['Cluster'], errors='coerce')

# # Load your data into a pandas DataFrame
# df = pd.read_csv('crypto_data_reduced.csv')
# df = df[['Index', 'Cluster']]

# Initialize the Dash app
app = dash.Dash(__name__)

# Function to create a histogram and a box plot for a given stock
# Function to create a histogram and a box plot for a given stock
def create_stock_distribution(stock_name, stock_data):

    print(stock_name, type(stock_data))
    # Create histogram
    histogram = go.Figure(data=[go.Histogram(x=stock_data, nbinsx=30, name=f'Histogram of {stock_name}')])
    histogram.update_layout(title_text=f'Histogram of {stock_name}', bargap=0.05)

    # Create box plot
    box_plot = go.Figure(data=[go.Box(y=stock_data, name=f'Box Plot of {stock_name}')])
    box_plot.update_layout(title_text=f'Box Plot of {stock_name}')

    mean_value = stock_data.mean()
    median_value = stock_data.median()
    mode_value = stock_data.mode().values[0]  # Mode can have multiple values; take the first one
    skewness = skew(stock_data)
    kurt = kurtosis(stock_data)

    # return histogram, box_plot, mean_value, median_value, mode_value, skewness, kurt
    return histogram, box_plot


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
        dcc.Link("Correlation", href="/correlation", className="nav-link"),
        dcc.Link("Distribution", href="/distribution", className="nav-link"),
        dcc.Link("Model Training", href="/training", className="nav-link"),
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
    elif pathname == "/correlation":
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
            [1.0, 'rgb(239,138,98)']
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
            # dcc.Graph(id='positive-correlations', figure=fig_positive),
            # dcc.Graph(id='negative-correlations', figure=fig_negative)
            # Create a container div with display flex to align items horizontally
            html.Div([
                # Place each graph in a div, setting the flex attribute for even spacing
                html.Div([dcc.Graph(figure=fig_positive)], style={'flex': '1'}),
                html.Div([dcc.Graph(figure=fig_negative)], style={'flex': '1'})
            ], style={'display': 'flex', 'flex-wrap': 'wrap'})
        ])
        # return html.H1("EDA Page Content")
    elif pathname == "/distribution":
        html.H2("Distribution Analysis of Stocks"),
        histogram_fig1, box_plot_fig1 = create_stock_distribution('BTC-USD', df_stocks['BTC-USD'])
        histogram_fig2, box_plot_fig2 = create_stock_distribution('ETH-USD', df_stocks['ETH-USD'])
        histogram_fig3, box_plot_fig3 = create_stock_distribution('USDT-USD', df_stocks['USDT-USD'])
        histogram_fig4, box_plot_fig4 = create_stock_distribution('MKR-USD', df_stocks['MKR-USD'])
        # histogram_fig1, box_plot_fig1, mean_value, median_value, mode_value, skewness, kurt = create_stock_distribution(
        #     'BTC-USD', df_stocks['BTC-USD'])
        # histogram_fig2, box_plot_fig2, mean_value, median_value, mode_value, skewness, kurt = create_stock_distribution(
        #     'ETH-USD', df_stocks['ETH-USD'])
        # histogram_fig3, box_plot_fig3, mean_value, median_value, mode_value, skewness, kurt = create_stock_distribution(
        #     'USDT-USD', df_stocks['USDT-USD'])
        # histogram_fig4, box_plot_fig4, mean_value, median_value, mode_value, skewness, kurt = create_stock_distribution(
        #     'MKR-USD', df_stocks['MKR-USD'])

        return html.Div([
            html.H1("Stock Distribution Analysis"),
            # Div for each chart, set to inline-block to place them on the same line
            html.Div([dcc.Graph(figure=histogram_fig1)], style={'display': 'inline-block', 'width': '25%'}),
            html.Div([dcc.Graph(figure=histogram_fig2)], style={'display': 'inline-block', 'width': '25%'}),
            html.Div([dcc.Graph(figure=histogram_fig3)], style={'display': 'inline-block', 'width': '25%'}),
            html.Div([dcc.Graph(figure=histogram_fig4)], style={'display': 'inline-block', 'width': '25%'}),
            # Div for each chart, set to inline-block to place them on the same line
            html.Div([dcc.Graph(figure=box_plot_fig1)], style={'display': 'inline-block', 'width': '25%'}),
            html.Div([dcc.Graph(figure=box_plot_fig2)], style={'display': 'inline-block', 'width': '25%'}),
            html.Div([dcc.Graph(figure=box_plot_fig3)], style={'display': 'inline-block', 'width': '25%'}),
            html.Div([dcc.Graph(figure=box_plot_fig4)], style={'display': 'inline-block', 'width': '25%'}),
        ])

    elif pathname == "/training":
        return html.H1("Models training")
    elif pathname == "/forecasting":
        return html.H1("Forecasting Page Content")
    else:
        # Default to home when nothing else is matched
        return html.H1("Home Page Content")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
