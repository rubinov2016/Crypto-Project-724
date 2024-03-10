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
from scipy.stats import shapiro, normaltest, anderson
import pandas as pd
# from LSTM_forecasting import LSTM_forecasting
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import load as joblib_load
import numpy as np
from datetime import timedelta

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

# # Load our data into a pandas DataFrame
# df = pd.read_csv('crypto_data_reduced.csv')
# df = df[['Index', 'Cluster']]

# Initialize the Dash app
app = dash.Dash(__name__)

#Charts for ML models
def Model_chart(df1, df2, metrics, name):
    return dcc.Graph(id='historical-forecast-chart',
          figure={
              'data': [
                  go.Scatter(x=df1.index, y=df1['Price'], mode='lines',
                             name='Historical', line=dict(color='blue')),
                  go.Scatter(x=df2.index, y=df2['Price'], mode='lines+markers',
                             name='Forecasted', line=dict(color='red'))
              ],
              'layout': go.Layout(
                  title=name + ' Historical vs. Forecasted Values',
                  xaxis={'title': 'Date'},
                  yaxis={'title': 'Value'},
                  hovermode='closest',
                  annotations=[
                      dict(
                          xref='paper', yref='paper',
                          x=0, y=1 - 0.05 * i,  # Adjust y position for each annotation
                          xanchor='left', yanchor='bottom',
                          text=f"{metric}: {value:.3f}" if metric != 'mape' else f"{metric}: {value:.2f}%",
                          font=dict(family='Arial', size=12),
                          showarrow=False,
                          align='left'
                      ) for i, (metric, value) in enumerate(metrics.items())
                  ],
                  margin=dict(b=100)
              )
          }
    )


# Function to create a histogram and a box plot for a given stock
def create_stock_distribution(stock_name, stock_data):

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
    stat_shapiro, p_value_shapiro = shapiro(stock_data)
    return histogram, box_plot, mean_value, median_value, mode_value, skewness, kurt, stat_shapiro, p_value_shapiro
    # return histogram, box_plot


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
        target_symbol = 'BTC-USD'  # Replace with our target stock symbol

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
        # Now use these in our heatmap creation
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

        # Now use these in our heatmap creation
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
        # histogram_fig1, box_plot_fig1 = create_stock_distribution('BTC-USD', df_stocks['BTC-USD'])
        # histogram_fig2, box_plot_fig2 = create_stock_distribution('ETH-USD', df_stocks['ETH-USD'])
        # histogram_fig3, box_plot_fig3 = create_stock_distribution('USDT-USD', df_stocks['USDT-USD'])
        # histogram_fig4, box_plot_fig4 = create_stock_distribution('MKR-USD', df_stocks['MKR-USD'])
        histogram_fig1, box_plot_fig1, mean_value1, median_value1, mode_value1, skewness1, kurt1, stat_shapiro1, p_value_shapiro1 = create_stock_distribution(
            'BTC-USD', df_stocks['BTC-USD'])
        histogram_fig2, box_plot_fig2, mean_value2, median_value2, mode_value2, skewness2, kurt2,stat_shapiro2, p_value_shapiro2 = create_stock_distribution(
            'ETH-USD', df_stocks['ETH-USD'])
        histogram_fig3, box_plot_fig3, mean_value3, median_value3, mode_value3, skewness3, kurt3,stat_shapiro3, p_value_shapiro3 = create_stock_distribution(
            'USDT-USD', df_stocks['USDT-USD'])
        histogram_fig4, box_plot_fig4, mean_value4, median_value4, mode_value4, skewness4, kurt4,stat_shapiro4, p_value_shapiro4 = create_stock_distribution(
            'MKR-USD', df_stocks['MKR-USD'])

        # Add annotations for the statistics
        histogram_fig1.add_annotation(
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Mean: {mean_value1:.2f}<br>Median: {median_value1:.2f}<br>Mode: {mode_value1:.2f}<br>Skewness: {skewness1:.2f}<br>Kurtosis: {kurt1:.2f}<br>Shap stat: {stat_shapiro1:.2f}<br>Shap p_val: {p_value_shapiro1:.2f}",
            showarrow=False,
            font=dict(size=12),
            align="center"
        )
        histogram_fig2.add_annotation(
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Mean: {mean_value2:.2f}<br>Median: {median_value2:.2f}<br>Mode: {mode_value2:.2f}<br>Skewness: {skewness2:.2f}<br>Kurtosis: {kurt2:.2f}<br>Shap stat: {stat_shapiro2:.2f}<br>Shap p_val: {p_value_shapiro2:.2f}",
            showarrow=False,
            font=dict(size=12),
            align="center"
        )
        histogram_fig3.add_annotation(
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Mean: {mean_value3:.2f}<br>Median: {median_value3:.2f}<br>Mode: {mode_value3:.2f}<br>Skewness: {skewness3:.2f}<br>Kurtosis: {kurt3:.2f}<br>Shap stat: {stat_shapiro3:.2f}<br>Shap p_val: {p_value_shapiro3:.2f}",
            showarrow=False,
            font=dict(size=12),
            align="center"
        )
        histogram_fig4.add_annotation(
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Mean: {mean_value4:.2f}<br>Median: {median_value4:.2f}<br>Mode: {mode_value4:.2f}<br>Skewness: {skewness4:.2f}<br>Kurtosis: {kurt4:.2f}<br>Shap stat: {stat_shapiro4:.2f}<br>Shap p_val: {p_value_shapiro4:.2f}",
            showarrow=False,
            font=dict(size=12),
            align="center"
        )

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
        df_LTSM_historical = pd.read_json('LSTM_historical.json', orient='index')
        df_LTSM_forecasted = pd.read_json('LSTM_forecast.json', orient='index')
        df_ARIMA_historical = pd.read_json('ARIMA_historical.json', orient='index')
        df_ARIMA_forecasted = pd.read_json('ARIMA_forecast.json', orient='index')
        df_Prophet_historical = pd.read_json('Prophet_historical.json', orient='index')
        df_Prophet_forecasted = pd.read_json('Prophet_forecast.json', orient='index')
        df_SVR_historical = pd.read_json('SVR_historical.json', orient='index')
        df_SVR_forecasted = pd.read_json('SVR_forecast.json', orient='index')
        LSTM_metrics = pd.read_json('LSTM_metrics.json', typ='series')
        ARIMA_metrics = pd.read_json('ARIMA_metrics.json', typ='series')
        return html.Div([
            Model_chart(df_LTSM_historical, df_LTSM_forecasted, LSTM_metrics, 'LSTM'),
            Model_chart(df_ARIMA_historical, df_ARIMA_forecasted, ARIMA_metrics, 'ARIMA'),
            Model_chart(df_Prophet_historical, df_Prophet_forecasted, LSTM_metrics, 'Prophet'),
            Model_chart(df_SVR_historical, df_SVR_forecasted, LSTM_metrics, 'SVR')
            ])
    else:
        # Default to home when nothing else is matched
        return html.H1("Home Page Content")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
