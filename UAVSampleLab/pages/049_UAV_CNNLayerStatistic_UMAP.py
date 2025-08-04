'''
Script to Explore the Spectral Information of Selected Input Layers & Calculate Dimension Reduction with (2D or 3D) UMAP
The Spectral Information extracted from the Dataset Matrix - Sample Composit - vie eCognition to csv file ('M:\_test2')

'''
import pathlib
import pandas as pd
import numpy as np
import umap
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import streamlit as st
import seaborn as sns
import altair as alt
#from altair_saver import save
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio
import vegafusion
# Required to enable VegaFusion transform extraction in Altair
alt.data_transformers.enable("vegafusion")

import config as cfg
import colors
pio.templates.default = "plotly_white"

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


# Define a dictionary with marker attributes for each trace
marker_styles = {
    'bare_soil': {'size': 12, 'symbol': 'circle'},
    'vital_crop': {'size': 10, 'symbol': 'diamond'},
    'vital_lodged_crop': {'size': 12, 'symbol': 'diamond-open'},
    'flowering_crop': {'size': 10, 'symbol': 'x'},
    'ripening_crop': {'size': 12, 'symbol': 'circle-open'},
    'dry_crop': {'size': 10, 'symbol': 'square'},
    'dry_lodged_crop': {'size': 12, 'symbol': 'square-open'},
    'weed_infestation': {'size': 10, 'symbol': 'cross'},
}

for key in marker_styles.keys():
    if key in colors.pattern_colors:
        marker_styles[key]['color'] = colors.pattern_colors[key]



def read_eCognition_statistics(
                                root,
                                #pattern=None
                                ):
    in_file = pathlib.Path(root)
    for i in in_file.iterdir():
        #if i.is_dir():
        #if str(i.name).startswith("{}".format(pattern)):

        crop_type_code, sensor, a, b = i.name.split('_')
        df = pd.read_csv(i.as_posix(), sep=';', decimal=' ', encoding='ANSI', engine='python', dtype='str',
                         skipinitialspace=True, na_values='undefined')
        df = pd.melt(df, id_vars=['inner_x', 'inner_y', 'level_name', 'class_name'], var_name='feature')
        df['sensor'] = sensor
        df['crop_type_code'] = crop_type_code
        return df




idata_path = f'{pathlib.Path(cfg.db_path).parent}\Statistic_eCognition\data4umap'

# --------------- Select the Crop Type & Read the Data -----------------
col01, col02, col03, col04= st.columns((2, 2, 2, 2))
with col01:
    CROP_TYPE_CODE = st.selectbox(label='Crop Type:', options=['WW', 'SG', 'WG', 'WR', 'KM', 'ZwFr'])
    CNN_Inputs = {
                      'WR': ['Mean A', 'Mean MExG', 'Mean NDVI', 'Mean U']
                    , 'SG': ['Mean EVI2', 'Mean ExR', 'Mean NDWI']
                    , 'KM': ['Mean BG', 'Mean GRVI', 'Mean NDSI', 'Mean NGRDI']
                    #, 'WW': ['Mean B5', 'Mean EVI2', 'Mean GRVI', 'Mean CIVE', 'Mean NDSI']
                    , 'WW': ['Mean B5', 'Mean BG', 'Mean CIVE', 'Mean EVI2']
                    }


    # Read  csv files with Spectral Information extracted from Sample Composit vie eCognition ('M:\_test2')
    df = read_eCognition_statistics(root=r'{}\{}'.format(idata_path, CROP_TYPE_CODE)
                                    #, pattern=CROP_TYPE_CODE
                                    )
    df = df[df['feature'].str.startswith("Mean ")]
    feature_list = df['feature'].unique().tolist()


# --------------- Select Features if required -----------------
#with col02:
#    ReferenceP = st.selectbox(label='Reference Panel:', options=['RP', 'noRP'])
#with col03:
#    SENSOR = st.selectbox(label='Sensor:', options=['CIR', 'RGB'])

col11, col12 = st.columns((5, 1))
with col11:
    FEATURE = st.multiselect(label='Feature:',
                            options=feature_list,
                            default=CNN_Inputs[CROP_TYPE_CODE],
                           )

df['value'] = df['value'].astype(float)
df = df[df['feature'].isin(FEATURE)].dropna()

range_=[]
domain_=[]
for i in df['class_name'].unique().tolist():
    range_.append(colors.pattern_colors[i])
    domain_.append(i)
df.feature = df.feature.str.split(' ').str[1]


#-------------------------------------------------------------------------------------------------------------------
if CROP_TYPE_CODE == 'WW':
    selection_ = {
            'bare_soil': ['B5', 'BG', 'EVI2'],
            'dry_crop': ['B5', 'BG', 'EVI2'],
            'dry_lodged_crop': ['B5', 'BG', 'EVI2'],
            'flowering_crop': ['EVI2',  'BG'],
            'vital_crop': ['BG', 'CIVE'],
            'vital_lodged_crop': ['BG', 'CIVE'],
            'weed_infestation': ['BG', 'CIVE', 'EVI2'],
            }

elif CROP_TYPE_CODE == 'WR':
    selection_ = {
            'bare_soil': ['A', 'MExG', 'U'],
            'dry_crop': ['A', 'MExG', 'NDVI', 'U'],
            'flowering_crop': ['MExG', 'NDVI', 'U'],
            'vital_crop': ['A', 'MExG', 'NDVI', 'U'],
            'vital_lodged_crop': ['A', 'MExG', 'NDVI', 'U'],
            'weed_infestation': ['A', 'MExG', 'NDVI', 'U'],
            }

if CROP_TYPE_CODE == 'SG':
    selection_ = {
            'bare_soil': ['EVI2', 'ExR'],
            'dry_crop': ['EVI2', 'ExR'],
            'dry_lodged_crop': ['EVI2', 'ExR'],
            'flowering_crop': ['EVI2', 'NDWI'],
            'vital_crop': ['EVI2', 'NDWI'],
            'vital_lodged_crop': ['EVI2', 'NDWI'],
            'weed_infestation': ['EVI2', 'NDWI'],
            }

if CROP_TYPE_CODE == 'KM':
    selection_ = {
            'bare_soil': ['BG', 'GRVI', 'NDSI', 'NGRDI'],
            'ripening_crop': ['NDSI'],
            'vital_crop': ['NDSI'],
            }

# -------------- This selects only Layers are defined for each POI and ovewrites the df -> defined in  selection_
# Initialize an empty DataFrame to store the concatenated results
concatenated_df = pd.DataFrame()
for cl, lyrs in selection_.items():
    print(cl, lyrs)
    query_string ="class_name == '{}' & feature in {}".format(cl, lyrs)
    concatenated_df = pd.concat([concatenated_df, df.query(query_string)], ignore_index=True)
df = concatenated_df
concatenated_df = None
#-------------------------------------------------------------------------------------------------------------------



with st.expander("Show Box-plot"):
    box_plot = alt.Chart(df
    ).transform_joinaggregate(
      # v_value='value',
       avg_value = 'mean(value)',
       std_value = 'stdev(value)',

       min_value = 'min(value)',
       max_value = 'max(value)',
       groupby = ['feature', 'class_name']
    ).transform_calculate(
        z_value="(datum.value - datum.avg_value) / datum.std_value",
        norm_value="(datum.value - datum.min_value) /(datum.max_value - datum.min_value)"
    ).mark_boxplot(size=27, extent=0.5, outliers={'size': 5}, opacity=0.8  # extent='min-max',.encode(
    ).encode(
        x=alt.X("class_name:N", title=None, axis=alt.Axis(labels=False, ticks=False, labelAngle=-30), scale=alt.Scale(padding=0.1), sort=None),
        y=alt.Y("norm_value:Q", axis=alt.Axis(ticks=True, domain=True), title=''),
        color=alt.Color('class_name:N').scale(domain=domain_, range=range_),
        column=alt.Column('feature:N', title='', header=alt.Header(labelFontSize=20))
    ).properties(
        width=30 * len(df['class_name'].unique()),
        height=35 * len(df['feature'].unique())
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16
    ).configure_facet(
        spacing=45,
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16
    ).configure_legend(
        titleFontSize=18,
        labelFontSize=20
        )
    st.write(box_plot)



with st.expander("Show values + std"):
    selection = alt.selection_point(fields=['class_name'], bind='legend')

    base = alt.Chart(df
    ).transform_joinaggregate(
      # v_value='value',
       avg_value = 'mean(value)',
       std_value = 'stdev(value)',

       min_value = 'min(value)',
       max_value = 'max(value)',
       groupby = ['feature', 'class_name']
    ).transform_calculate(
        z_value="(datum.value - datum.avg_value) / datum.std_value",
        norm_value="(datum.value - datum.min_value) /(datum.max_value - datum.min_value)",
    )

    points = base.mark_point(size=80,
        point={
          "filled": False,
          "fill": "white"
        }
    ).encode(
        x=alt.X("feature:O"),
        y=alt.Y("norm_value:Q", aggregate='mean'),
        color = alt.condition(selection,
                              alt.Color("class_name:N"),
                              alt.value('lightgray')),
        opacity=alt.condition(selection,
                              alt.value(1),
                              alt.value(0.05),
                              )
    ).properties(
        width=1800,
        height=400
    ).add_selection(
        selection
    )

    # generate the error bars
    #https://vega.github.io/vega-lite/docs/errorbar.html
    error_bars = base.mark_errorbar(extent='iqr' # extent='stdev' ['ci', 'iqr', 'stderr', 'stdev']
    , ticks=True
    ).encode(
      x=alt.X("feature:O"),
      y=alt.Y("norm_value:Q", scale=alt.Scale(zero=False)),
      color="class_name:N",
      strokeWidth=alt.value(1)
    )

    st.write(points + error_bars)

with st.expander("show the datframe"):
    st.write(df)


col11, _, _, _ = st.columns((2, 2, 2, 2))
with col11:
    SELECTION = st.multiselect(label='Group By:', options=['class_name', 'feature'], default=['class_name', 'feature'])

with st.expander("Show Density-plot (not the best representation for multidimensional data)"):

    # Custom x-axis tick values
    custom_ticks = [-1, 0, 1]

    density_plot1 = alt.Chart(df,
        width=250,
        height=250
    ).transform_joinaggregate(
      # v_value='value',
       #avg_value = 'mean(value)',
       #std_value = 'stdev(value)',
       min_value = 'min(value)',
       max_value = 'max(value)',
       groupby = SELECTION,
    ).transform_calculate(
        #z_value="(datum.value - datum.avg_value) / datum.std_value",
        norm_value="(datum.value - datum.min_value) /(datum.max_value - datum.min_value)"
    ).transform_density(
        #'value',
        'norm_value',
        groupby = SELECTION,
        as_=['value', 'density'],
        extent=[-0.2, 1.2],
        counts=True,
    ).mark_area(
    ).encode(
        #x="value:Q",
        #y='density:Q',
        alt.X('value:Q',
        axis=alt.Axis(values=custom_ticks,
            labelFontSize=28, labelColor='black', labelLimit=200, titleColor='black', titleFontSize=30, title=''),
            ),
        alt.Y('density:Q',
        axis=alt.Axis(labelFontSize=28, labelColor='black', labelLimit=200, titleColor='black', titleFontSize=30, title=''),
              ),#, stack='zero'
        alt.Color('class_name:N',
                  legend=alt.Legend(symbolLimit=500, columns=1, title='class'), #, columns=len(data['features'].unique())),
                  scale=alt.Scale(domain=domain_, range=range_)
                  ), #  scale=alt.Scale(scheme='category20b'),
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0.0)),
        tooltip='class_name:N',
        facet=alt.Facet('feature:N', columns=5, title='', header=alt.Header(labelFontSize=30))
    ).configure_facet(
        spacing=15,
    ).configure_axis(
        labelFontSize=28,
        titleFontSize=30
    ).configure_legend(
        titleFontSize=28,
        labelFontSize=30,
        labelLimit=0,  # Set labelLimit to 0 to show entire label names
        labelColor='black',
        titleColor='black',
        symbolSize=350
    ).resolve_scale(
        #x='independent',
        y='independent'

    ).add_selection(
        selection
    ).interactive(
    )
    st.write(density_plot1)


with st.expander("Show Density-plot after features"):

    SELECTION = ['class_name', 'feature']

        # Custom x-axis tick values
    custom_ticks = [-1, 0, 1]

    density_plot2 = alt.Chart(df,
        width=250,
        height=250
    ).transform_joinaggregate(
      # v_value='value',
       #avg_value = 'mean(value)',
       #std_value = 'stdev(value)',
       min_value = 'min(value)',
       max_value = 'max(value)',
        groupby = SELECTION,
    ).transform_calculate(
        #z_value="(datum.value - datum.avg_value) / datum.std_value",
        norm_value="(datum.value - datum.min_value) /(datum.max_value - datum.min_value)"
    ).transform_density(
        #'value',
        'norm_value',
        groupby = SELECTION,
        as_=['value', 'density'],
        extent=[-0.2, 1.2],
        counts=True,
    ).mark_area(
    ).encode(
        #x="value:Q",
        #y='density:Q',
        alt.X('value:Q',
        axis=alt.Axis(values=custom_ticks,
            labelFontSize=28, labelColor='black', labelLimit=200, titleColor='black', titleFontSize=30, title=''),
            ),
        alt.Y('density:Q',
        axis=alt.Axis(labelFontSize=28, labelColor='black', labelLimit=200, titleColor='black', titleFontSize=30, title=''
                      ),
              ),#, stack='zero'
        alt.Color('feature:N',
                  legend=alt.Legend(symbolLimit=500, columns=1, title='class'), #, columns=len(data['features'].unique())),
                  ), #  scale=alt.Scale(scheme='category20b'),
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0.0)),
        tooltip='class_name:N',
        facet=alt.Facet('class_name:N', columns=5, title='', header=alt.Header(labelFontSize=30))
    ).configure_facet(
        spacing=15,
    ).configure_axis(
        labelFontSize=28,
        titleFontSize=30
    ).configure_legend(
        titleFontSize=28,
        labelFontSize=30,
        labelLimit=0,  # Set labelLimit to 0 to show entire label names
        labelColor='black',
        titleColor='black',
        symbolSize=350
    ).resolve_scale(
        #x='independent',
        y='independent'
    ).add_params(
        selection
    ).interactive(
    )

    st.write(density_plot2)



step = 20
overlap = 1

# --------- generate the normalised data  with altair package
base = alt.Chart(df,
                height=step
).transform_joinaggregate(
    avg_value='mean(value)',
    std_value='stdev(value)',

    min_value='min(value)',
    max_value='max(value)',
    groupby=['feature', 'class_name']
).transform_calculate(
    z_value="(datum.value - datum.avg_value) / datum.std_value",
    norm_value="(datum.value - datum.min_value) /(datum.max_value - datum.min_value)"
)

# Calculate density of norm_value grouped by ['feature', 'class_name'] using transform_density
density_chart = base.transform_density(
    density='norm_value',
    groupby=['feature', 'class_name']
)


# get transformed data from the Altair chart
table = base.transformed_data()


#---------- this allows to select the equal amount of samples for each class_name (based on the smallest value of the class)

# Step 1: Find the maximum number of 'norm_values' for any class_name
max_rows = table[["feature", "class_name", "norm_value"]].groupby(["feature", "class_name"])['norm_value'].count().min()
st.write('amount of samples for each class_name (based on the smallest value of the class): ', max_rows)


sampled_data = []
for key, group in table[["feature", "class_name", "norm_value"]].groupby(["feature", "class_name"]):
    num_rows = len(group)
    sample_size = min(num_rows, max_rows)
    sample = group.sample(n=sample_size, random_state=1).reset_index(drop=True).reset_index(drop=False)
    sampled_data.append(sample)
sampled_df = pd.concat(sampled_data, ignore_index=True)
#st.write(group.sample(n=sample_size, random_state=1).reset_index(drop=True).reset_index(drop=False))



# Step 2: Create a new column to enumerate each 'class_name' within 'feature'
#df['class_index'] = df.groupby('feature').cumcount()
#table['norm_value'] = pd.to_numeric(table['norm_value'])
data = sampled_df[["index", "feature", "class_name", "norm_value"]].pivot(index=["class_name", "index"], columns="feature", values="norm_value").reset_index()

data["class_name_in"] = data["class_name"]
data = data.set_index(["class_name_in", "index"])

# reset the nan values to value = -0.1, otherwise it will not plot the 3 variable in 3D plot from pyplot
for columns_to_fill in table['feature'].unique().tolist():
    data[columns_to_fill] = data[columns_to_fill].fillna(-0.1)
    #data[columns_to_fill] = data[columns_to_fill].fillna(0)

with st.expander("Show DataFrame used for 3D plot"):
    st.write(data)


fig_ = px.scatter_matrix(data,
            dimensions=table['feature'].unique().tolist(),
            color='class_name'
            )


# Save the figure as an image (e.g., in PNG format)
pio.write_image(fig_, r'{}\_plots\scatter_matrix.png'.format(cfg.output_path))
#fig_.show()
#st.pyplot(fig_)



# this is the scatter matrix SPLOM (scatter plot matrix) with seaborn to represent 5D date
plot_seaborn_SPLOM = False
if plot_seaborn_SPLOM == True:
    plt.figure(figsize=(5, 5))
    #data['norm_value'] = pd.to_numeric(data['norm_value'])
    fig_x = sns.pairplot(data, hue='class_name', diag_kind="hist")
    #plt.savefig('scatterplot_matrix.png')
    st.pyplot(fig_x, use_container_width=True)




#----------------- this is the 3D plot for 3 selected Features
SIZE = 20

# Get a list of available columns for x, y, and z variables # Exclude "class_name" from the list of available columns
available_columns = [col for col in data.columns if col != 'class_name']

# Create multiselect widgets for x, y, and z variables
st.text('Select only 3 variables for 3D plot:')
variables = [col for i, col in enumerate(available_columns) if st.checkbox(f'{col}', value=i < 3)]


fig_3d_ = px.scatter_3d(data,
                        x=variables[0],
                        y=variables[1],
                        z=variables[2],
                        color='class_name',
                        symbol='class_name',
                        color_discrete_map=colors.pattern_colors,
                        opacity=0.5,
                        )

# Update the traces to customize the markers based on the dictionary
for species, attributes in marker_styles.items():
    fig_3d_.update_traces(
        selector=dict(name=species),
        marker=dict(size=attributes['size'], symbol=attributes['symbol'])
    )


# Update the layout to adjust tick label font size on all three axes
fig_3d_.update_layout(
    scene=dict(
        xaxis=dict(
            tickfont=dict(size=SIZE-5)  # Adjust the font size for X-axis tick labels
        ),
        yaxis=dict(
            tickfont=dict(size=SIZE-5)  # Adjust the font size for Y-axis tick labels
        ),
        zaxis=dict(
            tickfont=dict(size=SIZE-5)  # Adjust the font size for Z-axis tick labels
        )
    )
)

# Make the legend title and legend marker bigger
fig_3d_.update_layout(
    legend=dict(
        font=dict(size=20),  # Adjust the font size as needed for the title
        itemsizing='constant',  # To control the size of legend markers
        title=dict(text='class names', font=dict(size=20 + 5))  # Change the title and adjust its font size
    )
)

# Make the axis title labels bigger
fig_3d_.update_layout(
    scene=dict(
        xaxis_title_font=dict(size=SIZE),  # Adjust the font size for the x-axis title
        yaxis_title_font=dict(size=SIZE),  # Adjust the font size for the y-axis title
        zaxis_title_font=dict(size=SIZE)   # Adjust the font size for the z-axis title
    )
)

# Update the layout to adjust tick label font size
fig_3d_.update_xaxes(tickfont=dict(size=SIZE * 2))  # Adjust the X-axis tick label font size
fig_3d_.update_yaxes(tickfont=dict(size=SIZE * 2))  # Adjust the Y-axis tick label font size

# Set the size of the plot
fig_3d_.update_layout(
    width=1000,  # Adjust the width as needed
    height=1000  # Adjust the height as needed
)
# Add a title to the plot
fig_3d_.update_layout(title_text='3D plot of selected Layers - {}'.format(', '.join(variables)),
                      title_font=dict(size=SIZE + 2)
                      )

st.plotly_chart(fig_3d_, use_container_width=True)

# Save the plot as an HTML file
fig_3d_.write_html(r'{}\_plots\3d_scatter_plot_{}_{}_{}_{}.html'.format(cfg.output_path, CROP_TYPE_CODE, variables[0],variables[1], variables[2] ))
# Save the plot as an HTML file
pio.write_image(fig_3d_, r'{}\_plots\3d_scatter_plot_{}_{}_{}_{}.png'.format(cfg.output_path, CROP_TYPE_CODE, variables[0],variables[1], variables[2] ))


st.write('https://pair-code.github.io/understanding-umap/')
# Display a link to the saved HTML file
#st.markdown(f"**[Download the plot as HTML](3d_scatter_plot.html)**")



def update_chart(n_neighbors, random_state, to_dimension, metric, min_dist, save=True):

    reducer = umap.UMAP(random_state=random_state,
                        n_neighbors=n_neighbors,  # max_rows * len(data['class_name'].unique()) * 10000,
                        n_components=to_dimension,  # nimber of dimensions to be reduced to
                        metric=metric,
                        min_dist=min_dist,
                        )
    reducer.fit_transform(data_)
    embedding = reducer.transform(data_)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert (np.all(embedding == reducer.embedding_))

    #with st.expander("Show DataFrame with reduced dimenstion after UMAP method - used for 3D plot"):
        # st.write(data_)
    #    st.text('DF after dimension reduction')
    #    st.write(embedding.shape)
    #    st.write(embedding)

    if to_dimension == 3:
        SIZE = 40
        # Create a DataFrame from the 3D embedding
        df = pd.DataFrame(embedding, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
        # Add the 'class_name' labels back to the DataFrame
        df['class_name'] = class_names.values

        fig_3d = px.scatter_3d(df,
                                x='Dimension 1',
                                y='Dimension 2',
                                z='Dimension 3',
                                color='class_name',
                                symbol='class_name',
                                opacity=0.5,
                                color_discrete_map=colors.pattern_colors
                                )

        # Update the traces to customize the markers based on the dictionary
        for species, attributes in marker_styles.items():
            fig_3d.update_traces(
                selector=dict(name=species),
                marker=dict(size=attributes['size'],
                            symbol=attributes['symbol']
                            )
            )

        # Update the layout to adjust tick label font size on all three axes
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(
                    tickfont=dict(size=SIZE-10)  # Adjust the font size for X-axis tick labels
                ),
                yaxis=dict(
                    tickfont=dict(size=SIZE-10)  # Adjust the font size for Y-axis tick labels
                ),
                zaxis=dict(
                    tickfont=dict(size=SIZE-10)  # Adjust the font size for Z-axis tick labels
                )
            )
        )

        # Make the legend title and legend marker bigger
        fig_3d.update_layout(
            legend=dict(
                font=dict(size=24),  # Adjust the font size as needed for the title
                itemsizing='constant',  # To control the size of legend markers
                title=dict(text='class names', font=dict(size=20 + 5))  # Change the title and adjust its font size
            )
        )

        # Make the axis title labels bigger
        fig_3d.update_layout(
            scene=dict(
                xaxis_title_font_size=SIZE+10,  # Adjust the font size for the x-axis title
                yaxis_title_font_size=SIZE+10,  # Adjust the font size for the y-axis title
                zaxis_title_font_size=SIZE+10 # Adjust the font size for the z-axis title
            )
        )

        # Set the size of the plot
        fig_3d.update_layout(
            width=1800,  # Adjust the width as needed
            height=1000  # Adjust the height as needed
        )

        # Add a title to the plot
        fig_3d.update_layout(title_text='{} CNN Input Layers - {} - after UMAP Dimension Reduction'.format(colors.crop[CROP_TYPE_CODE][1], ', '.join(available_columns)),
                             title_font=dict(size=SIZE),
                             #title_font = dict(size=SIZE - 15)
                             )
        fig_3d.update_layout(margin=dict(l=100, r=100, t=100, b=100))
        fig_3d.update_layout(autosize=True)

        # Update the layout to adjust tick label font size
        #fig_3d.update_xaxes(tickfont=dict(size=SIZE*2))  # Adjust the X-axis tick label font size
       # fig_3d.update_yaxes(tickfont=dict(size=SIZE*2))  # Adjust the Y-axis tick label font size

        st.plotly_chart(fig_3d, use_container_width=True)
        if save==True:
            fig_3d.write_html(r'{}\_plots\{}d_umap_scatter_plot_{}_{}_{}_{}.html'.format(cfg.output_path, to_dimension, CROP_TYPE_CODE, variables[0],variables[1], variables[2]))
            # Save the plot as an HTML file
            pio.write_image(fig_3d, r'{}\_plots\{}d_scatter_plot_{}_{}_{}_{}.png'.format(cfg.output_path, to_dimension, CROP_TYPE_CODE, variables[0],variables[1], variables[2] ))



    if to_dimension == 2:

        # Perform dimensionality reduction using UMAP
        reducer = umap.UMAP(random_state=random_state,
                            n_neighbors=n_neighbors,
                            n_components=to_dimension,
                            metric=metric,
                            min_dist=min_dist)
        embedding = reducer.fit_transform(data_)

        # Define the range of cluster numbers you want to evaluate
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # Initialize an empty DataFrame to store silhouette scores for different cluster numbers
        results = pd.DataFrame(columns=['n_clusters', 'silhouette_score'])

        # Iterate over each cluster number
        for n_clusters in range_n_clusters:
            # Fit KMeans clustering model
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = clusterer.fit_predict(embedding)

            # Compute silhouette score
            silhouette_avg = silhouette_score(embedding, cluster_labels)

            # When adding a row
            new_row = pd.DataFrame([{'n_clusters': n_clusters, 'silhouette_score': silhouette_avg}])
            results = pd.concat([results, new_row], ignore_index=True)

        # Plot the UMAP projection using Plotly Express
        SIZE = 30
        df = pd.DataFrame(embedding, columns=['Dimension 1', 'Dimension 2'])
        df['class_name'] = class_names.values

        fig_2d = px.scatter(
            df,
            x='Dimension 1',
            y='Dimension 2',
            color='class_name',
            opacity=0.6,
            size_max=SIZE + 10,
            # title='UMAP projection'
            color_discrete_map=colors.pattern_colors,
        )


        # Update the layout and appearance of the UMAP plot
        #fig_2d.update_traces(marker=dict(symbol='circle', size=SIZE, line=dict(width=2, color='DarkSlateGrey')))
        for species, attributes in marker_styles.items():

            fig_2d.update_traces(
                selector=dict(name=species),
                marker=dict(size=attributes['size'],
                            symbol=attributes['symbol'],
                            color=attributes['color']
                )
            )

        # Display the UMAP plot using st.plotly_chart()
        #st.plotly_chart(fig_2d, use_container_width=True)

        # Create the plot
        fig_score = go.Figure()

        fig_score.add_trace(go.Scatter(
            x=results['n_clusters'],
            y=results['silhouette_score'],
            mode='lines+markers+text',
            text=results['silhouette_score'].round(2).astype(str),
            textposition="top center",
            marker=dict(color='black', size=round(SIZE/3)),
            line=dict(color='grey'),
            textfont=dict(
                color='black',  # Set text color to black
                size=round(SIZE-5),  # Adjust text size as needed
                family="Arial, bold"  # Make the font bold
            )
        ))

        # Update layout
        fig_score.update_layout(
            #title='Silhouette Scores for Different Numbers of Clusters',
            xaxis_title='Number of Clusters',
            yaxis_title='Silhouette Score',
            xaxis=dict(tickfont=dict(size=SIZE)),
            yaxis=dict(range=[0, 1],
                       tickvals=[0, 0.5, 1],  # Specify the positions for the ticks
                       #ticktext=['0', '0.5', '1'],  # Specify the text for each tick
                       tickfont=dict(size=SIZE)),
            xaxis_title_font_size=SIZE,
            yaxis_title_font_size=SIZE,
            width=850,
            height=350
        )

        LARGE_MARKER_SIZE = 120


        # Create a subplot with 2 rows and 1 column
        combined_fig = make_subplots(rows=2, cols=1,
                         #ubplot_titles = ("2D UMAP Dimension Reduction", "Silhouette Scores for Different Numbers of Clusters"),
                         row_heights=[0.7, 0.3],
                         vertical_spacing = 0.21  # Adjust spacing as needed
                                     )
        #st.plotly_chart(fig_score, use_container_width=True)

        # Dummy to make the legend markers Larger, but the legend traces are not clickable anymore
        for species, attributes in marker_styles.items():
            combined_fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                #selector=dict(name=species),
                marker=dict(size=LARGE_MARKER_SIZE, symbol=attributes['symbol'], color=attributes['color']),  # Large size for legend
                name=species,
                showlegend=True
            ))

        # Add the traces from the first figure (fig_2d) to the subplot
        for trace in fig_2d.data:
            trace.marker.size = 20
            trace.showlegend = False  # Hide these from the legend
            combined_fig.add_trace(trace, row=1, col=1)


        # Add the traces from the second figure (fig_score) to the subplot
        for trace in fig_score.data:
            #trace.marker.size = 30
            combined_fig.add_trace(trace, row=2, col=1)

        # Update x-axis and y-axis for the first subplot (fig_2d)
        combined_fig.update_xaxes(
            tickfont=dict(size=SIZE),
            title_text="Dimension 1",
            title_font=dict(size=SIZE),
            row=1, col=1,
        )
        combined_fig.update_yaxes(
            tickfont=dict(size=SIZE),
            row=1, col=1,
            title_text="Dimension 2",
            title_font=dict(size=SIZE),
        )

        # Update x-axis and y-axis for the second subplot (fig_score)
        combined_fig.update_xaxes(
            title="Number of Clusters",
            title_font=dict(size=SIZE),
            tickfont=dict(size=SIZE),
            row=2, col=1
        )

        if CROP_TYPE_CODE == 'SG':
            show_legend = True
        else:
            show_legend = False

        combined_fig.update_yaxes(
            title="Silhouette Score",
            range=[0, 1],
            tickvals=[0.0, 0.5, 1.0],
            ticktext=['0.0', '0.5', '1.0'],
            tickfont=dict(size=SIZE),
            title_font=dict(size=SIZE),
            row=2, col=1
        )
        LARGE_LEGEND_SIZE=28
        # Global layout updates (if needed)
        combined_fig.update_layout(
            height=1150, width=1000,  # Adjust total height and width as needed
            showlegend=show_legend,  # Set to False if you don't want to show the legend
            #legend=dict(font=dict(size=SIZE), itemsizing='constant', traceorder='normal'),
            legend=dict(
                x=1.05,
                y=1.05,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.5)',
                font=dict(size=LARGE_LEGEND_SIZE), itemsizing='constant', traceorder='normal'),
            #title=dict(text="2D UMAP Dimension Reduction", font=dict(size=SIZE - 15)),
            margin=dict(l=100, r=100, t=100, b=100)
        )

        combined_fig.data[-1].showlegend = False

        # Show the combined figure in Streamlit
        st.plotly_chart(combined_fig, use_container_width=True)

        # Save as HTML
        #combined_fig.write_html("combined_plot.html")

        # Save as PNG (requires 'kaleido')
        combined_fig.write_image(
            r'{}\_plots\{}d_umap_scatter_plot_SilhouetteScore_{}_{}_{}_{}_n{}_rs{}_md{}.png'.format(cfg.output_path,
                to_dimension, CROP_TYPE_CODE, variables[0], variables[1], variables[2], n_neighbors, random_state, min_dist_value))





# prepare data in wide format to transform data
class_names = data['class_name'].copy()
data_ = data.drop('class_name', axis=1)


st.title('Select Parameters for UMAP')
col41, col42, col43, col44 = st.columns((5, 5, 5, 5))
with col41:
    # Create a slider for selecting the category
    n_neighbors = st.slider('n_neighbors', min_value=len(selection_), max_value=int(max_rows), value=int(max_rows*0.4), step=2)
with col42:
    # Create a slider for selecting the category
    random_state = st.slider('random_state', min_value=int(max_rows/2), max_value=int(max_rows*2), value=max_rows, step=1)
with col43:
    # Create a slider for selecting the category
    min_dist_value = st.slider('min_dist_value', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
with col44:
    # Create a slider for selecting the category
    options = {2: 2, 3: 3}
    to_dimension_value = st.radio('dimension_value', list(options.keys()), format_func=lambda x: options[x])

st.write(to_dimension_value)

# Update the chart based on the selected category
update_chart(n_neighbors, random_state, to_dimension=to_dimension_value, metric='euclidean', min_dist=min_dist_value, save=False)



