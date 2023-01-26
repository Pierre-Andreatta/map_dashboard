# Librairies
import dash
from dash import dash_table
from dash import html, dcc, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objs as go

import pandas as pd
import geopandas as gpd
import json

import requests

from dash_style import style_header, style_cell_population, style_cell_co2, style_data_conditional, style_table



style_block = {'display': 'block'}
style_none = {'display': 'none'}


# =============================================================================
# Data Load
# =============================================================================

#https://www.kaggle.com/datasets/thedevastator/global-fossil-co2-emissions-by-country-2002-2022
co2_raw = pd.read_csv("Data/CO2_countries.csv").query("Total > 0")

#https://www.kaggle.com/datasets/juanmah/world-cities
cities_raw = pd.read_csv("Data/worldcities.csv").query("(population > 8000000)")

#https://www.kaggle.com/datasets/muhammedtausif/world-population-by-countries
population_raw = pd.read_csv("Data/world-population-by-country-2020.csv")

#https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/export/
gdf_raw = gpd.read_file("Data/world-administrative-boundaries.shp", encoding='UTF-8')

#https://gist.github.com/tadast/8827699
gps_countries_raw = pd.read_csv("Data/countries_codes_and_coordinates.csv")



# =============================================================================
# Data Prep
# =============================================================================

def gdf_prep(df):
    """
    Args:
        df (df): raw_df

    Returns:
        df: prep_df
    """
    
    gdf = df.copy()

    # Mismatch
    gdf.replace({'United States of America' : 'United States',
                'U.K. of Great Britain and Northern Ireland': 'United Kingdom',
                "Democratic People's Republic of Korea": 'South Korea',
                'Russian Federation': 'Russia',
                'Moldova, Republic of': 'Moldova',
                'Libyan Arab Jamahiriya': 'Libya',
                "Lao People's Democratic Republic": 'Laos',
                "Iran (Islamic Republic of)": "Iran",
                'Syrian Arab Republic': 'Syria',
                'Republic of Korea': 'North Korea',
                'United Republic of Tanzania': 'Tanzania'}, inplace= True)

    # Keep columns
    gdf = gdf[['name', 'continent', 'region', 'geometry']]
    gdf.rename(columns= {'name': 'Country'}, inplace= True)

    # Simplify polygons to optimize performance
    gdf.geometry = gdf.geometry.simplify(0.04)
    
    return gdf



def gps_countries_prep(df):
    """
    Args:
        df (df): raw_df

    Returns:
        df: prep_df
    """

    gps_countries = df.copy()

    # Remove " from df
    for i in list(gps_countries.columns):  
        gps_countries[i] = gps_countries[i].apply(lambda x: x.replace('"', ''))
        
    gps_countries['Numeric code'] = gps_countries['Numeric code'].astype(int)
    gps_countries['Latitude (average)'] = gps_countries['Latitude (average)'].astype(float)
    gps_countries['Longitude (average)'] = gps_countries['Longitude (average)'].astype(float)


    gps_countries.rename(columns= {'Alpha-3 code': 'iso3',
                                'Latitude (average)': 'latitude',
                                'Longitude (average)': 'longitude'}, inplace= True)

    gps_countries.iso3 = gps_countries.iso3.apply(lambda x: x.strip())

    # We want to keep unique Iso3 codes
    gps_countries.drop_duplicates(subset= 'iso3', keep= 'last', inplace= True)
    
    return gps_countries


def population_prep(df, gdf):
    """
    Args:
        df (df): raw_df
        gdf (_type_): _description_

    Returns:
        df: prep_df
    """
    
    population = df.copy()

    population.drop(columns='no', inplace= True)
    
    population.rename(columns={'Country (or dependency)': 'Country',
                                    'Population 2020': 'population_2020',
                                    'Yearly Change': 'yearly_change',
                                    'Net Change': 'net_change',
                                    'Density  (P/Km²)': 'density',
                                    'Land Area (Km²)': 'area',
                                    'Migrants (net)': 'migrants',
                                    'Fert. Rate': 'fert_rate',
                                    'Med. Age': 'med_age',
                                    'Urban Pop %': 'urban_pop',
                                    'World Share': 'world_share'}, inplace= True)

    # Drop NaN
    population = population.dropna(subset= 'migrants').copy()

    # Drop NaN as string
    population.drop(list(population.query("urban_pop.str.contains('N.A')").index), inplace= True)

    # Feature format and type
    import re
    population.population_2020 = population.population_2020.apply(lambda x: int(re.sub("[^0-9]", "", x)))
    population.net_change = population.net_change.apply(lambda x: int(re.sub("[^0-9]", "", x)))
    population.area = population.area.apply(lambda x: int(re.sub("[^0-9]", "", x)))
    population.migrants = population.migrants.apply(lambda x: int(re.sub("[^0-9]", "", x)))

    population.yearly_change = population.yearly_change.apply(lambda x: float(x[:-1]))
    population.urban_pop = population.urban_pop.apply(lambda x: float(x[:-1]))
    population.world_share = population.world_share.apply(lambda x: float(x[:-1]))

    population.density = population.density.apply(lambda x: float(x.replace(',', '.')))
    population.fert_rate = population.fert_rate.apply(lambda x: float(x.replace(',', '.')))

    population.med_age = population.med_age.astype(int)


    population.replace({'DR Congo': 'Democratic Republic of the Congo',
                        'Czech Republic (Czechia)': 'Czech Republic'}, inplace= True)

    df_geo = pd.merge(gdf,
                    population,
                    on= 'Country',
                    how= 'inner').set_index("Country").fillna(0)
    
    return df_geo


def co2_prep(df, gps_countries):

    co2 = df.copy()
    co2.rename(columns= {'ISO 3166-1 alpha-3': 'iso3'}, inplace= True)
    co2.fillna(0, inplace= True)

    df_co2_geo = pd.merge(gps_countries[['iso3', 'latitude', 'longitude']],
                        co2,
                        on= 'iso3',
                        how= 'inner')

    df_co2_geo.Country.replace({'USA': 'United States',
                                'Viet Nam': 'Vietnam'}, inplace= True)

    df_co2_geo_poly = pd.merge(gdf,
                            df_co2_geo,
                            on= 'Country',
                            how= 'inner')

    #Max Year
    df_co2_geo_poly_2021 = df_co2_geo_poly.query(f"Year == {df_co2_geo_poly.Year.max()}")

    co2_2021_melt = pd.melt(df_co2_geo_poly_2021,
                            id_vars= ['Country'],
                            value_vars= list(df_co2_geo_poly_2021.columns[9:-1]),
                            var_name= 'max_co2',
                            value_name= 'total')

    co2_2021_max = co2_2021_melt.sort_values('total', ascending= False)
    co2_2021_max.drop_duplicates(subset='Country', keep='first', inplace= True)

    df_co2_geo_poly_2021 = pd.merge(df_co2_geo_poly_2021,
                                    co2_2021_max[['Country', 'max_co2']],
                                    on= 'Country',
                                    how= 'inner')

    #Decade
    bins = pd.cut(df_co2_geo['Year'], bins=range(df_co2_geo.Year.min(), df_co2_geo.Year.max(), 10))
    df_co2_decade = df_co2_geo.groupby([bins, 'Country']).sum(numeric_only=True).drop(columns='Year').reset_index()
    df_co2_decade.Year = df_co2_decade.Year.apply(lambda x: str(str(x)[1:5]))

    df_co2_decade.drop(columns=['latitude', 'longitude'], inplace= True)

    df_co2_decade = pd.merge(df_co2_decade,
                            df_co2_geo_poly_2021[['Country', 'region', 'continent','latitude', 'longitude']],
                            how= 'inner',
                            on= 'Country'
                            )
    df_co2_decade.rename(columns= {'Year': 'Decade'}, inplace= True)

    df_co2_decade.Decade = df_co2_decade.Decade.astype(str)
    
    result = {'co2_2021_max': df_co2_geo_poly_2021,
              'df_co2_decade': df_co2_decade}
    
    return result



# =============================================================================
# Data Export
# =============================================================================

gdf = gdf_prep(gdf_raw)
gps_countries = gps_countries_prep(gps_countries_raw)
population = population_prep(population_raw, gdf)
co2 = co2_prep(co2_raw, gps_countries)
co2_2021_max = co2['co2_2021_max']
df_co2_decade = co2['df_co2_decade']

# =============================================================================
# Tools
# =============================================================================


# Dictionary for filtering the map and data 

map_region = {'Europe': ["continent == 'Europe'",
                         {'lat': 54.5,'lon': 13.4},
                         2,2]}

map_region.update({'Asia': ["continent == 'Asia'",
                            {'lat': 28.4,'lon': 84.6},
                            2]})

map_region.update({'Oceania': ["continent == 'Oceania'",
                               {'lat': -27.8,'lon': 173.2},
                               2]})

map_region.update({'Africa': ["continent == 'Africa'",
                              {'lat': 4.7,'lon': 18.1},
                              2]})

map_region.update({'Northern America': ["(continent == 'Americas') & (region in ['Northern America', 'Central America'])",
                                        {'lat': 55,'lon': -98.5},
                                        1.6]})

map_region.update({'South America': ["(continent == 'Americas') & (region == 'South America')",
                                     {'lat': -23.1,'lon': -68.6},
                                     2]})

map_region.update({'World': [f"continent in {list(population.continent.unique())}",
                             {'lat': 46.6,'lon': 2.8},
                             0.8]})


# Variables and hover_templates

custom_population = ['Country', 'population_2020', 'yearly_change', 'density', 'urban_pop']

hovertemplate_density='<b>%{customdata[0]}</b><extra></extra>'+'<br>%{customdata[3]}'
hovertemplate_urban_pop='<b>%{customdata[0]}</b><extra></extra>'+'<br>%{customdata[2]}'

custom_max_co2 = ['Country', 'Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other']
hovertemplate_max_co2='<b>%{customdata[0]}</b><extra></extra>'+'<br>%{customdata[1]} (Giga-tonnes)'




# =============================================================================
# Fonctions selected country template
# =============================================================================

def country_template(flag_link, name, currencie, currencie_symbol, language, capital, area, population):
    """
    Template for a country

    Returns:
        dbc.Card: template
    """
    
    area = f'{int(area):,}'.replace(',', ' ')
    population = f'{int(population):,}'.replace(',', ' ')
    
    template = dbc.Card([
        dbc.Row([
            dbc.Col(
                html.H1(f'{name}', className= 'custom_text h1_text'),
                width={'size':'auto'}
                ),
            dbc.Col(
                html.Img(src= flag_link, 
                        className = 'flag'
                        )
                )
            ],
            align=True,
            justify= 'left'),
        html.Div(),
        html.Div([
            dcc.Markdown(f'''
                         **Currencie** : {"&emsp;"*2} {currencie} ({currencie_symbol})  
                         **Language** : {"&emsp;"*3} {language}  
                         **Capital** : {"&emsp;"*4} {capital}   
                         **Area** : {"&emsp;"*7} {area} km²  
                         **Population** : &emsp; {population}''',
                         className= 'custom_text cor_text')

        ])
    ],className= 'card_content_style'
                        )
    
    return template



# =============================================================================
# Fonctions selected countries tables
# =============================================================================


def pop_selected_countries_table(points):
    """
    Args:
        points (list): selectedData from figure

    Returns:
        df
    """
    
    df = pd.DataFrame(columns=['Country', 'Population', 'Yearly change', 'Density', 'Urban pop.'])
    
    for i in range(len(points)):
        df.loc[i] = points[i]['customdata']
        
    return df.sort_values('Population', ascending= False)



def co2_selected_countries_table(points):
    """
    Args:
        points (list): selectedData from figure

    Returns:
        df
    """
    
    df = pd.DataFrame(columns=['Country', 'Total Co2', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other', 'Max Co2', 'Max value'])
    
    for i in range(len(points)):
        df.loc[i] = points[i]['customdata']
        
    df[['Total Co2', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other', 'Max Co2', 'Max value']] = df[['Total Co2', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other', 'Max Co2', 'Max value']].round(2)
        
    return df.sort_values('Total Co2', ascending= False)


# =============================================================================
# Plots Fonctions
# =============================================================================

def choro_map(df, center, zoom, feature, color_continuous_scale, custom_data, hovertemplate):
    """
    choropleth_mapbox from plotly.express
    
    Args:
        df (df): 
        center (dict): {'lat': -23.1,'lon': -68.6}
        zoom (in): 
        feature (str): feature to display
        color_continuous_scale (str):
        custom_data (list): features to store in figure
        hovertemplate (str): template

    Returns:
        figure
    """
    
    data = df.copy()
    
    fig = px.choropleth_mapbox(data, 
                               geojson=data.geometry, 
                               locations=data.index, 
                               color= feature,
                               color_continuous_scale= color_continuous_scale,
                               mapbox_style="carto-positron",
                               zoom=  zoom, 
                               center = center,
                               opacity=0.35,
                               height= 500,
                               custom_data= custom_data,
                               # width=700,
                               #title= f"{feature} par pays",
                               color_discrete_map={
                                                    'Coal': '#383838',
                                                    'Cement': '#f7b748',
                                                    'Oil': '#db5f40',
                                                    'Gas': '#4498c2',
                                                    'Other': '#a1a1a1',
                                                    'Flaring': '#9fbf3f'
                                                }
                               )
    
    fig.update_traces(hovertemplate=hovertemplate)

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20),
                      clickmode='event+select')
    
    fig.layout.update(dragmode='select')
    
    
    return fig



def scatter_map(df, center, zoom):
    """
    scatter_mapbox from plotly.express

    Args:
        df : data
        center (dict): {'lat': -23.1,'lon': -68.6}
        zoom (in): 

    Returns:
        figure
    """
    
    data = df.copy()
    
    fig = px.scatter_mapbox(data,
                            lat= 'latitude',
                            lon= 'longitude',
                            color= 'Total',
                            size= 'Total',
                            size_max= 100,
                            mapbox_style= 'carto-positron',
                            zoom=  zoom, 
                            center = center,
                            #color_continuous_scale= 'bluered',
                            color_continuous_scale= ['#4498c2', '#db5f40', '#383838'],
                            range_color= [data.Total.min(), data.Total.max()],
                            color_discrete_map={
                                'Coal': '#383838',
                                'Cement': '#f7b748',
                                'Oil': '#db5f40',
                                'Gas': '#4498c2',
                                'Other': '#a1a1a1',
                                'Flaring': '#9fbf3f'
                                },
                            height= 600,
                            # width= 700,
                            animation_frame='Decade')
    
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=5))
    
    fig.layout.update(dragmode='select')
    
    #fig['layout']['updatemenus'][0]['pad']=dict(r= 10, t= 150)
    fig['layout']['sliders'][0]['pad']=dict(r= 40, t= 10,)
    
    return fig



def sunburst(df, feature, feature_color, reversed_colors, custom_data, hovertemplate):
    """
    sunburst from plotly.express
    
    Args:
        df (df): 
        feature (str): feature to display
        feature_color (str): 
        reversed_colors (Bool): 
        custom_data (list): features to store in figure
        hovertemplate (str): template

    Returns:
        figure
    """
    
    if reversed_colors:
        color_continuous_scale = px.colors.diverging.RdYlGn[::-1]
    else:
        color_continuous_scale = 'RdYlGn'
    
    data = df[['continent', 'region', 'Country', feature, feature_color]]
    
    if data.continent.nunique() > 1:
        path= ['continent', 'region', 'Country']
    else:
        path= ['region', 'Country']
    
    fig = px.sunburst(data,
                      path= path,
                      values = feature,
                      color= feature_color,
                      color_continuous_scale= color_continuous_scale,
                      maxdepth=2,
                      custom_data= custom_data,
                      
                      #hover_data={"id": False},
                    #   height= 600,
                    #   width=700
                      )
    
    fig.update_traces(hovertemplate=hovertemplate)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=5))
    
    return fig



def bar_plot(df, hovertemplate):
    """
    bar from plotly.express
    
    Args:
        df (df): 
        hovertemplate (str): template

    Returns:
        figure
    """
    
    data = df.groupby('Decade').sum(numeric_only=True).reset_index().sort_values('Decade')
    
    fig = px.bar(data,
              x= 'Decade',
              y= ['Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other'],
              color_discrete_map={
                     'Coal': '#383838',
                     'Cement': '#f7b748',
                     'Oil': '#db5f40',
                     'Gas': '#4498c2',
                     'Other': '#a1a1a1',
                     'Flaring': '#9fbf3f'
              },
              height=400,
              template= 'simple_white'
              )
    
    hovertemplate ='<b>Giga-tonnes</b> : %{y}'+'<br><b>Decade</b> : %{x}'

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    fig.update_traces(hovertemplate=hovertemplate)

    return fig



# =============================================================================
# col_options
# =============================================================================

region_options = [dict(label=x, value=x) for x in list(map_region.keys())]


# =============================================================================
# Components
# =============================================================================

dd_region = [html.Div("Region", 
                      className = 'dropdown-label'
                      ), 
             html.Div(children=[
                 dcc.Dropdown(id="dd_region", 
                              placeholder="Region...",
                              value=region_options[-1]['value'],
                              options=region_options, 
                              multi=False,
                              clearable=False,
                              style = {'color':'#383838'}
                              )]
                      )
             ]

population_button = dbc.Button('Population', 
                               className= 'nav_button', 
                               id= 'population_button')

co2_button = dbc.Button('Co2', 
                        className= 'nav_button', 
                        id= 'co2_button')


# Graphs

choro_co2_max = dcc.Graph(id='choro_co2_max',
                          config = {
                              # 'scrollZoom': False,
                              # 'displayModeBar': True,
                              # 'editable': True,
                              # 'showLink':False,
                              'displaylogo': False
                              }
                          )

choro_density = dcc.Graph(id='choro_density',
                          config = {
                              # 'scrollZoom': False,
                              # 'displayModeBar': True,
                              # 'editable': True,
                              # 'showLink':False,
                              'displaylogo': False
                              }
                          )


choro_urban_pop = dcc.Graph(id='choro_urban_pop',
                            config = {
                                # 'scrollZoom': False,
                                # 'displayModeBar': True,
                                # 'editable': True,
                                # 'showLink':False,
                                'displaylogo': False
                                }
                            )

hist_co2_type_decade = dcc.Graph(id='hist_co2_type_decade',
                                 config = {
                                     # 'scrollZoom': False,
                                     # 'displayModeBar': True,
                                     # 'editable': True,
                                     # 'showLink':False,
                                     'displaylogo': False
                                     }
                                 )

scatter_co2_decade = dcc.Graph(id='scatter_co2_decade',
                               config = {
                                   # 'scrollZoom': False,
                                   # 'displayModeBar': True,
                                   # 'editable': True,
                                   # 'showLink':False,
                                   'displaylogo': False
                                   }
                               )

sunburst_population = dcc.Graph(id='sunburst_population',
                                config = {
                                    # 'scrollZoom': False,
                                    # 'displayModeBar': True,
                                    # 'editable': True,
                                    # 'showLink':False,
                                    'displaylogo': False
                                    }
                                )



pop_selected_countries_tables = dash_table.DataTable(id= 'pop_selected_countries_tables',
                                                     export_format= 'xlsx',
                                                     export_headers= 'display',
                                                     fixed_rows= {'headers': True},
                                                     style_header= style_header,
                                                     style_cell= style_cell_population,
                                                     sort_action= 'native',
                                                     merge_duplicate_headers= True,
                                                     style_data_conditional= style_data_conditional,
                                                     style_table= style_table
                                                     )


co2_selected_countries_tables = dash_table.DataTable(id='co2_selected_countries_tables',
                                                     export_format= 'xlsx',
                                                     export_headers= 'display',
                                                     fixed_rows= {'headers': True},
                                                     style_header= style_header,
                                                     style_cell= style_cell_co2,
                                                     sort_action= 'native',
                                                     merge_duplicate_headers= True,
                                                     style_data_conditional= style_data_conditional,
                                                     style_table= style_table
                                                     )




population_content = [
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Div("Population density",
                         className= 'custom_text graph_title'),
                choro_density]
                ),
            width={'size':6},
            className= 'grid_div_top_left'
            ),
        dbc.Col(
            html.Div([
                html.Div("Urban population rate",
                         className= 'custom_text graph_title'),
                choro_urban_pop],
                className= 'grid_div_top_right'),
            width={'size':6}
            )
        ]
    ),
    dbc.Row([
        dbc.Col([
            html.Div(id='population_selected_countrie_div',
                     className= 'selected_countrie_div'),
            html.Div(pop_selected_countries_tables,
                     id='pop_selected_countries_tables_div',
                     className= 'selected_countrie_div',
                     style= {'display': 'none'}),
            html.Div()
            ],
            width={'size':6}
            ),
        dbc.Col(
            html.Div([
                html.Div("Population and growth rate",
                         className= 'custom_text graph_title'),
                sunburst_population],
                className= 'grid_div_bot_right'),
            width={'size':6}
            )
    ])   
]


co2_content = [
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Div("Primary source of CO2",
                         className= 'custom_text graph_title'),
                choro_co2_max],
                className= 'grid_div_top_left'),
            width={'size':6}
            ),
        dbc.Col(
            html.Div([
                html.Div("CO2 emissions per decade",
                         className= 'custom_text graph_title'),
                scatter_co2_decade,
                html.Div()],
                className= 'grid_div_top_right'),
            width={'size':6}
            )
        ]
    ),
    dbc.Row([
        dbc.Col([
            html.Div(id='co2_selected_countrie_div',
                     className= 'selected_countrie_div'),
            html.Div(co2_selected_countries_tables,
                     id='co2_selected_countries_tables_div',
                     className= 'selected_countrie_co2_div',
                     style= {'display': 'none'})
            ],
            width={'size':6}
            ),
        dbc.Col(
            html.Div([
                html.Div(id= 'hist_co2_type_decade_title',
                         className= 'custom_text graph_title'),
                hist_co2_type_decade],
                className= 'grid_div_bot_right'),
            width={'size':6}
            )
    ])
    
    
]


# =============================================================================
# App 
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# =============================================================================
# Layout
# =============================================================================
app.layout = html.Div(children=[

    dbc.Row([
            dbc.Col(style={'background-color': '#383838',
                           'height': '100vh'}, 
                    children=[html.Div(dd_region),
                                html.Br(),
                                population_button,
                                html.Br(),
                                co2_button],
                    width={'size':1},
                    className = "fixed-top"),
            dbc.Col(
                dcc.Loading(
                    id='intermediate-value_loading',
                    children =[
                        html.Div(id= 'content_div'),
                        dcc.Store(id='population_store'),
                        dcc.Store(id='co2_2021_store'),
                        dcc.Store(id='geo_data_store'),
                        dcc.Store(id='co2_decade_store'),
                        dcc.Store(id='cities_store')
                    ],
                    className= 'loading_page_gif',
                    type='graph',
                    fullscreen=True,)
                ,
            width={'size':11, 'offset':1})
        ])  
    
        
    # Children end
    ],
    # Container end
    )


# =============================================================================
# Callback
# =============================================================================


# Button Layout

@app.callback([Output('content_div', 'children'),
               Output('population_button', 'className'),
               Output('co2_button', 'className')],
              [Input('population_button', 'n_clicks'),
               Input('co2_button', 'n_clicks')])
def content_button_cb(population_button, co2_button):
    """
    Retrieve the most recently activated button to display its associated content and apply its button style
    """
    
    button_clicked = ctx.triggered_id
    if button_clicked == 'population_button':
        return population_content, 'nav_button nav_button_pressed', 'nav_button'
    elif button_clicked == 'co2_button':
        return co2_content, 'nav_button', 'nav_button nav_button_pressed'
    else:
        return None, 'nav_button', 'nav_button'
    


# Data Store

@app.callback(Output('population_store', 'data'),
              [Input('dd_region', 'value')])
def population_store_cb(dd_region):
    
    region = str(dd_region)
    
    population_reg = population.query(map_region[region][0]).reset_index()
    
    print("population_store Json OK")
    return json.dumps(population_reg.__geo_interface__)



@app.callback(Output('co2_2021_store', 'data'),
              [Input('dd_region', 'value')])
def co2_2021_store_cb(dd_region):
    
    region = str(dd_region)
    
    co2_2021_max_reg = co2_2021_max.query(map_region[region][0])
    co2_2021_max_reg['max_value'] = co2_2021_max_reg.apply(lambda x: x[x['max_co2']], axis= 1)
    
    print("co2_2021_store Json OK")
    return json.dumps(co2_2021_max_reg.__geo_interface__)



@app.callback(Output('co2_decade_store', 'data'),
              [Input('dd_region', 'value')])
def co2_decade_store_cb(dd_region):
    
    region = str(dd_region)
    
    df_co2_decade_reg = df_co2_decade.query(map_region[region][0])
    df_co2_decade_reg = df_co2_decade_reg.query("Total > 0").sort_values('Decade').reset_index(drop= True)
    df_co2_decade_json = df_co2_decade_reg.to_json(orient='split', date_format='iso')
    
    print("co2_decade_json_store Json OK")
    return json.dumps(df_co2_decade_json)



@app.callback(Output('geo_data_store', 'data'),
              [Input('dd_region', 'value')])
def geo_data_store_cb(dd_region):
    
    region = str(dd_region)
    
    geo_data = {'zoom':  map_region[region][2],
                'center': map_region[region][1]}
        
    print("geo_data_store Json OK")
    return json.dumps(geo_data)



@app.callback(Output('cities_store', 'data'),
              [Input('dd_region', 'value')])
def cities_store_cb(dd_region):
    
    region = str(dd_region)
    
    gdf = co2_2021_max.query(map_region[region][0])
    
    cities_list = list(gdf.iso3.unique())
    selected_cities = cities_raw.query("iso3 == @cities_list")
    mark_lat = selected_cities.lat
    mark_lon = selected_cities.lng
    mark_text = selected_cities.city
    population = selected_cities.population
    
    data = {'mark_lat': list(mark_lat),
            'mark_lon': list(mark_lon),
            'mark_text': list(mark_text),
            'population': list(population)}  
    
    print("cities_store Json OK")
    
    return json.dumps(data)




# =============================================================================
# Population Plots 
# =============================================================================

@app.callback(Output('sunburst_population', 'figure'),
              [Input('population_store', 'data'),
               Input('choro_density', 'selectedData'),
               Input('choro_urban_pop', 'selectedData')])
def sunburst_population_cb(data, selectedData_density, selectedData_urban_pop):
    """
    Sunburst chart that is based on the selected data
    """
    
    # Last input                
    button_clicked = ctx.triggered_id

    if button_clicked == 'choro_density':
        try:
            selectedData = selectedData_density['points']    
        except:
            # Unselect on map
            selectedData = None
             
    elif button_clicked == 'choro_urban_pop':
        try:
            selectedData = selectedData_urban_pop['points']
        except:
            # Unselect on map
            selectedData = None
             
    else:
        #No country has been selected yet
        selectedData = None




    if selectedData:
        if len(selectedData) > 1:
            # Multiple countries selected
            selected_countries = [selectedData[i]['customdata'][0] for i in range(len(selectedData))]
            gdf = gpd.GeoDataFrame.from_features(json.loads(data)).query("Country == @selected_countries")
        else:
            # One country selected
            gdf = gpd.GeoDataFrame.from_features(json.loads(data))
    else:
        # No country selected
        gdf = gpd.GeoDataFrame.from_features(json.loads(data))
    
    custom_population = ['population_2020', 'yearly_change']
    hovertemplate_density='<b>Population</b> : %{customdata[0]:,}<extra></extra>'+'<br><b>Growth rate</b> : %{customdata[1]:.2f}%'
    
    fig = sunburst(gdf, 'population_2020', 'yearly_change', True, custom_population, hovertemplate_density)
    
    fig.update_layout(coloraxis_colorbar=dict(title='Growth rate'))
    
    return fig



@app.callback(Output('choro_density', 'figure'),
              [Input('population_store', 'data'),
               Input('geo_data_store', 'data')])
def chro_density_cb(data, geo_data_json):
    
    geo_data = json.loads(geo_data_json)
    
    gdf = gpd.GeoDataFrame.from_features(json.loads(data))
    
    custom_population = ['Country', 'population_2020', 'yearly_change', 'density', 'urban_pop']
    hovertemplate_density='<b>%{customdata[0]}</b><extra></extra>'+'<br>Density : %{customdata[3]} hab/km'
    
    fig = choro_map(gdf, geo_data['center'], geo_data['zoom'],'density', ["#faef8c", "#f24418", "#0a0200"], 
                    custom_population, hovertemplate_density)
    
    fig.update_layout(coloraxis_colorbar=dict(title='Density'))
    
    return fig


@app.callback(Output('choro_urban_pop', 'figure'),
              [Input('population_store', 'data'),
               Input('geo_data_store', 'data'),
               Input('cities_store', 'data')])
def choro_urban_pop_cb(data, geo_data_json, cities):
    
    geo_data = json.loads(geo_data_json)
    
    gdf = gpd.GeoDataFrame.from_features(json.loads(data))
    
    custom_population = ['Country', 'population_2020', 'yearly_change', 'density', 'urban_pop']
    hovertemplate_urban_pop='<b>%{customdata[0]}</b><extra></extra>'+'<br>Urban population : %{customdata[4]}%'
        
    fig = choro_map(gdf, geo_data['center'], geo_data['zoom'],'urban_pop', 'RdYlGn_r', 
                    custom_population, hovertemplate_urban_pop)
    
    fig.update_layout(coloraxis_colorbar=dict(title='Urban Pop.'))
    
    
    lat = list(json.loads(cities)['mark_lat'])
    lon = list(json.loads(cities)['mark_lon'])
    population = list(json.loads(cities)['population'])
    size= [i/1300000 for i in population]
    
    hovertemplate = '<b>%{text}</b><extra></extra>'+'<br>%{customdata:,d} (pop)' 
    
    fig.add_trace(go.Scattermapbox(
        customdata=population,
        lat=lat,
        lon=lon,
        mode="markers+text",
        # marker = {'size':15}, px.colors.diverging.RdYlGn
        marker={'color': '#36394d', 'opacity': 0.7, 'size': size},
        # textfont=dict(size=16, color='black'),
        text=json.loads(cities)['mark_text'],
        # textposition='top right',
        # hovertext=True,
        # meta='text',
        hovertemplate= hovertemplate,
        showlegend = False
    ))
    
    return fig


@app.callback([Output('population_selected_countrie_div', 'children'),
               Output('pop_selected_countries_tables_div', 'style')],
              [Input('choro_density', 'selectedData'),
               Input('choro_urban_pop', 'selectedData'),
               Input('dd_region', 'value')])
def population_selected_countrie_cb(selectedData_density, selectedData_urban_pop, dd_region):
    """
    Returns the template or table based on the selection
    """
    
    button_clicked = ctx.triggered_id
    if button_clicked == 'choro_density':
        selectedData = selectedData_density
    elif button_clicked == 'choro_urban_pop':
        selectedData = selectedData_urban_pop
    elif button_clicked == 'dd_region':
        return 'Please click on a state', style_none
    else:
        selectedData = None
        
        
    if selectedData:
        
        # If one country is selected, return country template
        if len(selectedData['points']) == 1:
            
            country = selectedData['points'][0]['customdata'][0]
            response = requests.get(f"https://restcountries.com/v2/name/{country}".replace('United States', 'USA'))
            data = response.json()
            
            if country == 'India':
                data.pop(0)
                
            flag_link = data[0]['flag']
            name = data[0]['name']
            currencie = data[0]['currencies'][0]['name']
            currencie_symbol = data[0]['currencies'][0]['symbol']
            language = data[0]['languages'][0]['name']
            capital = data[0]['capital']
            area = data[0]['area']
            population = data[0]['population']
            template = country_template(flag_link, name, currencie, currencie_symbol, language, capital, area, population)

            return template, style_none
        
        # If multiple countries are selected, return countries table
        else:
            return None, style_block
        
    else:
        return 'Please click on a state', style_none
    
    
    
@app.callback([Output('pop_selected_countries_tables', 'data'),
               Output('pop_selected_countries_tables', 'columns')],
              [Input('choro_density', 'selectedData'),
               Input('choro_urban_pop', 'selectedData')])
def pop_selected_countries_tables_cb(selectedData_density, selectedData_urban_pop):
    """
    Returns table based on the selection

    Returns:
        table: data
        table: columns
    """
    
    button_clicked = ctx.triggered_id
    if button_clicked == 'choro_density':
        selectedData = selectedData_density
    elif button_clicked == 'choro_urban_pop':
        selectedData = selectedData_urban_pop
    else:
        selectedData = None
        

    if selectedData:
        if len(selectedData['points']) > 1:
            
            df = pop_selected_countries_table(selectedData['points'])
            
            #Columns
            columns=[{'id': c,
                      'name': c, 
                      'type': 'numeric', 
                      'format': dict(specifier=',', 
                                  locale=dict(group=' ', grouping=[3]))
                      } for c in df.columns]
    
            return df.to_dict('records'), columns
        
        else:
            return pd.DataFrame({'None': ['None']}).to_dict('records'), [{'id': 'None', 'name': 'None'}]
    
    else:  
        return pd.DataFrame({'None': ['None']}).to_dict('records'), [{'id': 'None', 'name': 'None'}]




# =============================================================================
# Co2 Plots 
# =============================================================================



@app.callback([Output('hist_co2_type_decade', 'figure'),
               Output('hist_co2_type_decade_title', 'children')],
              [Input('co2_decade_store', 'data'),
               Input('choro_co2_max', 'selectedData'),
               Input('scatter_co2_decade', 'selectedData')])
def hist_co2_type_decade_cb(data, selectedData_co2_max, selectedData_co2_decade):
    
    button_clicked = ctx.triggered_id
    
    if button_clicked == 'choro_co2_max':
        try:
            selectedData = selectedData_co2_max['points']    
        except:
             selectedData = None
             
    elif button_clicked == 'scatter_co2_decade':
        try:
            selectedData = selectedData_co2_decade['points']
        except:
             selectedData = None
             
    else:
        selectedData = None


    if selectedData:
        selected_countries = [selectedData[i]['customdata'][0] for i in range(len(selectedData))]
        df = pd.read_json(json.loads(data), orient='split').query("Country == @selected_countries")
        if len(selected_countries) > 1:
            country_title = "Sources of CO2 emissions per decade"
        else :
            country_title = f"Sources of CO2 emissions per decade ({selected_countries[0]})"
    else:
        df = pd.read_json(json.loads(data), orient='split')   
        country_title = "Sources of CO2 emissions per decade"
    
   
    hovertemplate ='<b>%{customdata[0]}</b><extra></extra>'+'<br>%{customdata[8]} : %{customdata[9]:,d} Gt' 
     
    fig = bar_plot(df, hovertemplate)
    
    fig.update_layout(legend_title="Co2 source",
                      plot_bgcolor='#F0F2F3'
                      )
    
    return fig, country_title



@app.callback(Output('choro_co2_max', 'figure'),
              [Input('co2_2021_store', 'data'),
               Input('geo_data_store', 'data')])
def chro_co2_max_cb(data, geo_data_json):
    
    gdf = gpd.GeoDataFrame.from_features(json.loads(data))#.set_index('Country')
    
    geo_data = json.loads(geo_data_json)
    
    custom_max_co2 = ['Country', 'Total', 'Coal', 'Oil', 'Gas', 
                      'Cement', 'Flaring', 'Other', 'max_co2', 'max_value']
    
    hovertemplate_max_co2 ='<b>%{customdata[0]}</b><extra></extra>'+'<br>%{customdata[8]} : %{customdata[9]:,d} Gt'
    
    fig = choro_map(gdf, geo_data['center'], geo_data['zoom'],'max_co2', 'RdYlGn', custom_max_co2, hovertemplate_max_co2)
    
    fig.update_layout(legend_title="Co2 Max")
    #                   clickmode='event+select')
    
    return fig



@app.callback(Output('scatter_co2_decade', 'figure'),
              [Input('co2_decade_store', 'data'),
               Input('geo_data_store', 'data')])
def scatter_co2_decade_cb(data, geo_data_json):
    
    geo_data = json.loads(geo_data_json)
    
    df = pd.read_json(json.loads(data), orient='split')
    
    
    fig = scatter_map(df, geo_data['center'], geo_data['zoom'])
    
    fig.update_layout(coloraxis_colorbar=dict(title='Total Co2'))
    
    return fig



@app.callback([Output('co2_selected_countrie_div', 'children'),
               Output('co2_selected_countries_tables_div', 'style')],
              [Input('choro_co2_max', 'selectedData'),
               Input('scatter_co2_decade', 'selectedData'),
               Input('dd_region', 'value')])
def co2_selected_countrie_cb(selectedData_co2_max, selectedData_co2_decade, dd_region):
    
    button_clicked = ctx.triggered_id
    if button_clicked == 'choro_co2_max':
        selectedData = selectedData_co2_max
    elif button_clicked == 'scatter_co2_decade':
        selectedData = selectedData_co2_decade
    elif button_clicked == 'dd_region':
        return 'Please click on a state', style_none
    else:
        selectedData = None
    
    if selectedData:

        if len(selectedData['points']) == 1:
            
            country = selectedData['points'][0]['customdata'][0]
            response = requests.get(f"https://restcountries.com/v2/name/{country}".replace('United States', 'USA'))
            data = response.json()
            
            if country == 'India':
                data.pop(0)
            
            flag_link = data[0]['flag']
            name = data[0]['name']
            currencie = data[0]['currencies'][0]['name']
            currencie_symbol = data[0]['currencies'][0]['symbol']
            language = data[0]['languages'][0]['name']
            capital = data[0]['capital']
            area = data[0]['area']
            population = data[0]['population']
            template = country_template(flag_link, name, currencie, currencie_symbol, language, capital, area, population)

            return template, style_none
        
        else:
            return None, style_block
        
    else:
        return 'Please click on a state', style_none
    
    
    
@app.callback([Output('co2_selected_countries_tables', 'data'),
               Output('co2_selected_countries_tables', 'columns')],
              [Input('choro_co2_max', 'selectedData')])
def co2_selected_countries_tables_cb(selectedData):
    
    if selectedData:
        if len(selectedData['points']) > 1:
            
            variables = ['Country', 'Total Co2', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other', 'Max Co2']
            df = co2_selected_countries_table(selectedData['points'])[variables]
            
            #Columns
            columns=[{'id': c,
                    'name': c, 
                    'type': 'numeric', 
                    'format': dict(specifier=',', 
                                locale=dict(group=' ', grouping=[3]))
                    } for c in df.columns]
    
            return df.to_dict('records'), columns
                  

    else:
        return pd.DataFrame({'None': ['None']}).to_dict('records'), [{'id': 'None', 'name': 'None'}]



# =============================================================================
# Run App
# =============================================================================

if __name__ == '__main__':
    app.run_server(host= '0.0.0.0',port='8050', debug=False)