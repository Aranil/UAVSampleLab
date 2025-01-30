

import streamlit as st
import altair as alt



import colors
import config as cfg
import dashboard_utils as dutils
from dbflow.src import db_utility as dbu




alt.data_transformers.enable('default', max_rows=None)

col01, col02 = st.columns((1, 1))
with col01:
    CROP_TYPE = st.selectbox(label='Select AOI:', options=['WW', 'SG', 'WG', 'WR', 'ZwFr', 'KM']) #, 'WG', 'WR', 'ZwFr', 'KM'
with col02:
    JMD = st.slider(label='JMD threshold:', min_value=1.0, max_value=2.0, value=1.9, step=0.1)
#THRESHOLD = st.slider(label='feature threshold:', min_value=1, max_value=20, value=16, step=1)
st.write('')

# AND refpanel == 'RP' # only for Corn is noRP

if CROP_TYPE == 'KM':
    R_Panel = 'noRP'
else:
    R_Panel = 'RP'



sql_filter = f"""

SELECT * FROM (
			SELECT
				feature, 
				jmd, 
				class1, class2, 
				refpanel
				--COUNT(jmd) as count_jmd

			FROM
				uavfeaturejmd
				
			WHERE 
				jmd >= {JMD}
			AND (class1  LIKE '{CROP_TYPE}%' OR class2 LIKE 'UNK%' AND class1 LIKE 'UNK%' OR class2 LIKE '{CROP_TYPE}%' )
			AND 
                (feature NOT LIKE '%DSM' OR feature NOT LIKE '%DSM%')
                AND feature NOT LIKE 'GLCM%' AND feature NOT LIKE 'GLDV%' AND feature NOT LIKE 'Radius%'
                AND feature NOT LIKE 'Elliptic%' AND feature NOT LIKE 'Roundness%' AND feature NOT LIKE 'Density%'
                AND refpanel == '{R_Panel}'
			GROUP BY
				feature,
				class1, class2
				
			ORDER BY 
				--COUNT(jmd) DESC,
				feature 
				) 
--WHERE count_jmd >= 17
;
"""

#AND refpanel == '{R_Panel}' # this was deleted only for Corn

number_of_samples = {
                    'WW': 89,
                    'SG': 32,
                    'WR': 26,
                    'KM': 10
                    }

if CROP_TYPE == 'KM':
    Proportion_for_Feature_Importance = 0.20
elif CROP_TYPE == 'WR':
    Proportion_for_Feature_Importance = 0.15
elif CROP_TYPE == 'WW':
    Proportion_for_Feature_Importance = 0.15
else:
    Proportion_for_Feature_Importance = 0.15



# Calculate SELECT_THRESH

#st.write(SELECT_THRESH)

number_of_samples = {
                    'WW': 66 - 20,
                    'WR': 60 - 20,
                    'SG': 27 - 20,
                    'KM': 18 - 18
                    }

SELECT_THRESH = number_of_samples[CROP_TYPE]
SELECT_THRESH=0
# Print the result
st.write(number_of_samples[CROP_TYPE])
st.write(SELECT_THRESH)

#SELECT_THRESH = 18 # only fo Corn

df = dbu.query_sql(sql_filter, db_engine=cfg.dbarchive.archive.engine)

highlight = alt.selection_multi(fields=['count'], bind='legend')
click = alt.selection_multi(encodings=['x', 'y'], bind='legend')

df2 = dutils.feature_layer_split(df)['df_main']
df1 = dutils.feature_layer_split(df)['df_texture']
source = dutils.feature_layer_split(df)['df']

# Step 1: Aggregate and filter the data in Pandas
aggregated_data = df2.groupby(['layers', 'feature', 'feature_level']).size().reset_index(name='count')
filtered_data = aggregated_data[aggregated_data['count'] > SELECT_THRESH]
unique_features = filtered_data['feature'].unique()

st.write(df2)

df2['features'] = df2['features'].replace('quantile[50]', 'Median')


# Create a custom color scale transitioning from green to blue
custome_virdis = [

               '#482878',
               '#3E4989',
               '#31688E',
               '#26828E',
               '#1F9E89',
               '#35B779',
               '#6DCC5B',
               #'#B4DD2C',
               #'#FDE725'
            ]
custom_scale = alt.Scale(
    range=custome_virdis[::-1]  # Green to blue
)

df2 = df2.query("layers != 'DSM'"
                "& features != 'mode[Median]' "
                "& layers != 'index' "
                )

#df2 = df2.drop('feature_level', axis=1)

st.write(df2)

heatmap1 = alt.Chart(df2
).transform_aggregate(
    count='count():Q',
    groupby=['layers', 'features', 'feature_level']
    #groupby=['layers', 'features']
).transform_filter(
    alt.datum.count > SELECT_THRESH  # Filtering the counts to be greater than 60
).mark_rect(stroke='lightgrey', strokeWidth=0.1
).encode(
    x=alt.X('layers:O',# bin=False, #sort=None,
            sort=alt.EncodingSortField(field='count', op='max', order='descending'),
            axis=alt.Axis(labelFontSize=28, labelColor='black', labelLimit=300, titleColor='black', titleFontSize=30,
                          labelAlign='right', labelPadding=10),
            #title = alt.Title(text='X Axis Label', fontSize=16, color='black')
            #title = 'layer',
            title=''
            #title = 'Layers of {}'.format(colors.crop_types[CROP_TYPE])
            ),
    y=alt.Y('features:O', #sort=alt.EncodingSortField(field='features', order='ascending'),
            sort=alt.EncodingSortField(field='count', op='max', order='descending'),
            axis=alt.Axis(labelFontSize=28, labelColor='black', labelLimit=300, titleColor='black', titleFontSize=30,
                          labelAlign='right', labelPadding=10),
            #title = alt.Title(text='X Axis Label', fontSize=16, color='black')
            #title='feature'
            title=''
            ),
    color=alt.Color('count:Q',
                    scale=custom_scale,
                    #scale=alt.Scale(scheme='viridis'),
                    legend=alt.Legend(titleFontSize=32, titleColor='black', labelFontSize=30, title='Count', labelColor='black')), # scheme='redblue',

    opacity=alt.condition(highlight, alt.value(1), alt.value(0.2)),
    tooltip=['features', 'layers', 'count:Q', 'feature_level:O'],
    #tooltip=['features', 'layers', 'count:Q'],
).properties(
    width=1650,
    height=470,
    #width=1650, #/ 1.1, # only for Corn
    #height=470/ 1.23, # only for Corn
    title=alt.TitleParams(text='{}'.format(colors.crop_types[CROP_TYPE]), fontSize=22, align='center')
#).configure_legend(titleColor='black', titleFontSize=14
).add_selection(
    highlight,
    #click,
)


heatmap1.configure(
    font='Helvetica',
    axis=alt.AxisConfig(labelFont='Helvetica', titleFont='Helvetica'),
    legend=alt.LegendConfig(labelFont='Helvetica', titleFont='Helvetica'),
    title=alt.TitleConfig(font='Helvetica')
)
st.write(heatmap1)



# Remove the first three characters (i.e., 'SG_') from 'class1' and 'class2' columns
df2['class1'] = df2['class1'].str.slice(3)
df2['class2'] = df2['class2'].str.slice(3)


# Create the heatmap
heatmap_texture = alt.Chart(df2).transform_aggregate(
    count='count(jmd):Q',
    groupby=['class1', 'class2']
).encode(
    alt.X('class1:O', title=''),
    alt.Y('class2:O', title=''),
    opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
    tooltip=['class1', 'class2', 'count:Q']
).mark_rect(stroke='lightgrey', strokeWidth=1).encode(
    color=alt.Color('count:Q', scale=alt.Scale(scheme='greenblue')
                    , legend=alt.Legend(title='Count')
                    )
).add_selection(
    highlight,
 #   selector,
    click
)

heatmap_texture.configure(
    font='Helvetica',
    axis=alt.AxisConfig(labelFont='Helvetica', titleFont='Helvetica'),
    legend=alt.LegendConfig(labelFont='Helvetica', titleFont='Helvetica'),
    title=alt.TitleConfig(font='Helvetica')
)

st.write(heatmap_texture)



heatmaps = alt.hconcat(heatmap1,

                     heatmap_texture
                     )#.resolve_scale(
                     #   x='shared',
                        #y='shared'
                    # )



# Extract the transformed data from the chart
#chart_dict = heatmap1.to_dict()
# Find the transformed data in the chart's specification
#transformed_data = chart_dict['datasets'][list(chart_dict['datasets'].keys())[0]]
# Convert the transformed data to a Pandas DataFrame
#transformed_df = pd.DataFrame(transformed_data)





# Step 1: Extract unique values from 'features' column
#unique_features = transformed_df['feature'].unique()

# Step 2: Write the unique values to a text file
with open(r'F:\RCM\sarbian\xx_03_processing\rcm-dashboard\_temp\unique_features_{}.txt'.format(CROP_TYPE), 'w') as f:
    for feature in unique_features:
        f.write(f"{feature}\n")







col21, col22 = st.columns((1, 5))
#with col21:
#    JMD = st.slider(label='JMD threshold:', min_value=1.0, max_value=2.0, value=1.9, step=0.1)



#slider = alt.binding_range(min=0, max=15, step=1, name='cutoff:')
#selector = alt.selection_single(name="SelectorName", fields=['count'],
#                                bind=slider, init={'count(jmd):Q': 3})


# Configure common options
rect = alt.Chart(source
).transform_aggregate(
    count='count():O',
    groupby=['class1', 'class2']
).encode(
    alt.X('class1:O', scale=alt.Scale(paddingInner=0)),
    alt.Y('class2:O', scale=alt.Scale(paddingInner=0)),
    #color=alt.condition(highlight, 'count()', alt.value('lightgray')),
    opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
    tooltip=['class1', 'class2', 'count()']
).mark_rect(stroke='lightgrey', strokeWidth=1
).encode(
        color=alt.Color('count(jmd):O', scale=alt.Scale(scheme='greenblue')),
        #order=alt.Order('count(jmd):O', sort='ascending')
        #legend=alt.Legend(direction='horizontal'))
).add_selection(
    highlight,
 #   selector,
    click
)



# ---- PLot Heat plots
heatmap3 = alt.Chart(source
).mark_rect(stroke='lightgrey', strokeWidth=1).encode(
    #x=alt.X('layers:O', sort=None, title='layers/degree'),
    #y=alt.Y('features:O', sort=alt.EncodingSortField(field='features', order='ascending')),
    x=alt.X('layers:O', sort=alt.EncodingSortField(field='layers', op='count', order='descending'), title='layers/degree'),
    y=alt.Y('features:O', sort=alt.EncodingSortField(field='features', op='count', order='ascending')),
    color=alt.Color('count(jmd):Q', scale=alt.Scale(scheme='greenblue')), # scheme='redblue',
   # order=alt.Order('count(jmd):Q', sort='descending'),
    #opacity=alt.condition('count(jmd):Q' < selector.cutoff, alt.OpacityValue(1), alt.OpacityValue(0.1)),
    tooltip=['features', 'layers',  'count(jmd)', 'feature_level:N']
).transform_filter(
    click,
).properties(
    width=600,
    height=450
).add_selection(
    highlight
)


plot = alt.hconcat(
    heatmap3,
    rect,
)

plot_text = alt.vconcat(plot)

st.write(plot_text)


FEATURE = st.selectbox(label='Feature:', options=source['features'].unique().tolist())
print(FEATURE[:-1])


sql_filter2 = f"""
                SELECT
                    uavfeaturestatistic.class_name,
                    uavfeaturestatistic.feature,
                    ROUND(uavfeaturestatistic.value, 2) as value,
                    --substr(class_name, 1, instr(class_name, '_')-1) crop_type_code, 
                    substr(uavfeaturestatistic.class_name, instr(uavfeaturestatistic.class_name, '_')+1, 2) phenogroup, 
                
                    uavfeaturestatistic.fid,
                    uavfeaturestatistic.aoi,
                    uavfeaturestatistic.bbch,
                    uavfeaturestatistic.crop_type_code,
                    uavfeaturestatistic.uav_date,
                    --uavfeaturestatistic.refpanel,
                    --jmd_table.feature,
                    --COUNT(uavfeaturejmd.class_name),
                    jmd_table.jmd,
                    jmd_table.class,
                    jmd_table.class_to_compare,
                    jmd_table.refpanel
                    --COUNT(jmd_table.feature) as count_feature
                
                FROM 
                    uavfeaturestatistic
                
                LEFT JOIN 
                        (
                                    SELECT
                                            uavfeaturejmd.feature, 
                                            uavfeaturejmd.jmd, 
                                            uavfeaturejmd.class1 as class,
                                            uavfeaturejmd.class2 as class_to_compare,
                                            uavfeaturejmd.refpanel
                                        FROM
                                            uavfeaturejmd
                                        --WHERE 
                                            --uavfeaturejmd.class1  LIKE 'WW%'
                                            --OR (uavfeaturejmd.class2 LIKE 'WW%' AND uavfeaturejmd.class1 LIKE 'UNK%')
                
                                        --UNION ALL
                
                                        --SELECT
                                        --	uavfeaturejmd.feature, 
                                        --	uavfeaturejmd.jmd, 
                                        --	uavfeaturejmd.class2 as class,
                                        --	uavfeaturejmd.class1 as class_to_compare,
                                        --	uavfeaturejmd.refpanel
                                        --FROM
                                        --	uavfeaturejmd
                                        ----WHERE 
                                        --	--uavfeaturejmd.class2 LIKE 'WW%'
                                        --	--OR (uavfeaturejmd.class1 LIKE 'WW%' AND uavfeaturejmd.class2 LIKE 'UNK%')
                                        WHERE uavfeaturejmd.feature LIKE '{FEATURE[:-1]}%'
                                            ) as jmd_table
                
                ON (uavfeaturestatistic.feature == jmd_table.feature AND uavfeaturestatistic.refpanel == jmd_table.refpanel AND uavfeaturestatistic.class_name == jmd_table.class)
                
                WHERE 
                    jmd_table.jmd IS NOT NULL
                AND 
                    uavfeaturestatistic.refpanel != 'noRP'
                AND 
                    jmd_table.jmd >= {JMD}
                AND 
                    uavfeaturestatistic.crop_type_code LIKE '{CROP_TYPE}%' 
                    AND (uavfeaturestatistic.crop_type_code LIKE '{CROP_TYPE}%' OR jmd_table.class_to_compare LIKE '{CROP_TYPE}%' OR jmd_table.class_to_compare LIKE 'UNK%')
                    
                    --OR uavfeaturestatistic.crop_type_code LIKE 'UNK%'
                
                --GROUP BY 
                    --jmd_table.feature, 
                    --jmd_table.class, 
                    --uavfeaturestatistic.crop_type_code
                
                ORDER BY 
                    uavfeaturestatistic.crop_type_code
                ;
                """

sql_filter2 = f"""
------------------------------------------------------------------
---  get values of feature where jmd > 1.3
------------------------------------------------------------------
SELECT
	uavfeaturestatistic.class_name,
	ROUND(uavfeaturestatistic.value, 2) as value,
	--substr(class_name, 1, instr(class_name, '_')-1) crop_type_code, 
	substr(uavfeaturestatistic.class_name, instr(uavfeaturestatistic.class_name, '_')+1, 2) phenogroup, 
	jmd_table.feature, 
	jmd_table.class, 
	jmd_table.jmd
FROM 
	uavfeaturestatistic
LEFT JOIN ( 
					SELECT
						uavfeaturejmd.feature, 
						uavfeaturejmd.class1 as class,
						--class2, 
						uavfeaturejmd.jmd
					FROM
						uavfeaturejmd
					----WHERE 
					--	uavfeaturejmd.jmd > 1.3
					--AND uavfeaturejmd.class1 LIKE 'SG%'

						
					UNION
					SELECT
						uavfeaturejmd.feature, 
						--class1,
						uavfeaturejmd.class2 as class, 
						uavfeaturejmd.jmd
					FROM
						uavfeaturejmd
					--WHERE 
						--uavfeaturejmd.jmd > 1.3
					--AND uavfeaturejmd.class1 LIKE 'SG%'
					ORDER BY 
						uavfeaturejmd.jmd DESC,
						uavfeaturejmd.class1, 
						uavfeaturejmd.class2,
						uavfeaturejmd.feature ) as jmd_table

ON (uavfeaturestatistic.class_name == jmd_table.class AND uavfeaturestatistic.feature == jmd_table.feature)
WHERE 
	(uavfeaturestatistic.class_name LIKE '{CROP_TYPE}%' OR uavfeaturestatistic.class_name LIKE 'UNK%')
AND jmd_table.jmd NOT NULL
AND jmd_table.jmd > {JMD}
AND jmd_table.feature LIKE '{FEATURE[:-1]}%'


ORDER BY
--	uavfeaturestatistic.refpanel,
uavfeaturestatistic.class_name, 
uavfeaturestatistic.feature
;
"""
dt = cfg.query_sql(sql_filter2, db=cfg.dbarchive3.archive.engine)
data = dutils.feature_layer_split(dt)['df']
print(data.columns)
print(data.head())


#data['index'] = data['layers']
#data = data.set_index('index')
#diff = set(sorter.keys()).difference(set(data.index))
#data = data.loc[set(sorter.keys()) - diff]

box_plot = alt.Chart(data
#).transform_filter(
# 'datum.value != null'
#).transform_aggregate(
    #count='count()',
    #v_value = 'value',
#    avg_value='mean(value)',
#    std_value='stdev(value)',
#    groupby=['feature']
).transform_joinaggregate(
  # v_value='value',
   avg_value = 'mean(value)',
   std_value = 'stdev(value)',

   min_value = 'min(value)',
   max_value = 'max(value)',
   #groupby = ['feature', 'class_name']
    groupby = ['feature']

).transform_calculate(
    z_value="(datum.value - datum.avg_value) / datum.std_value",
    norm_value="(datum.value - datum.min_value) /(datum.max_value - datum.min_value)"

).mark_boxplot(size=30, extent=0.5, outliers={'size': 5}, opacity=0.5 # extent='min-max',.encode(
).encode(
    x=alt.X('layers:N', title=None, axis=alt.Axis(labels=True, ticks=True), scale=alt.Scale(padding=1), sort=None),
    y=alt.Y('z_value:Q', scale=alt.Scale(domain=[-15, 15])),
   #y=alt.Y('norm_value:Q'),
    color=alt.Y('class_name:N', scale=alt.Scale(zero=False)),
    #row = 'phenogroup'
    #column='feature'
    #color=alt.condition(selector, 'class_name:N', alt.ColorValue('gray'))
#).add_selection(
#        brush
).properties(width=1500, height=800).configure_axis(
    labelFontSize=16,
    titleFontSize=16
)

st.write(box_plot)


ranked_text = alt.Chart(source).mark_text(align='right'
                                        ).transform_aggregate(
                                        #most_count='argmax(count(jmd))',
                                        count='count(jmd)',
                                        #features='features:O',
                                        #layers='layers:O',
                                        groupby=['class1', 'class2', 'features', 'layers'] #
                                       # ).transform_calculate(
                                        #feature='datum.most_count.features',
                                        #layer='datum.most_count.layers',
                                        #count='datum.count',
                                       # class11='datum.most_count.class1',
                                       # class22='datum.most_count.class2'
                                        ).transform_window(
                                            row_number='row_number()',
                                            #rank='rank(most_count.features)',
                                            #rank_lyr='rank(layers)',
                                            #sort=[alt.SortField('most_count.features', order='descending')]
                                        ).transform_filter(
                                            # 'datum.rank == max(datum.rank)'
                                            alt.datum.count <= 2
                                        ).transform_filter(
                                            click
                                        ).encode(
                                        x=alt.X(),
                                        y=alt.Y('features:O', axis=None, sort=alt.EncodingSortField(op='count', order='descending'))#, sort=alt.EncodingSortField('layers')
                                        )

#https://stackoverflow.com/questions/64902970/argmax-aggregation-and-select-n-biggest-values-in-altair


#layers_count = ranked_text.encode(text='count(layers):O').properties(title=alt.TitleParams(text='layers', align='right'))
layers = ranked_text.encode(text='layers:O').properties(title=alt.TitleParams(text='layers', align='right'))
feature = ranked_text.encode(text='features:O').properties(title=alt.TitleParams(text='features', align='right'))
#count = ranked_text.encode(text='rank:O').properties(title=alt.TitleParams(text='rank', align='right'))
count = ranked_text.encode(text='count:O').properties(title=alt.TitleParams(text='count', align='right'))
#class1 = ranked_text.encode(text='class11:O').properties(title=alt.TitleParams(text='class1', align='right'))
#class2 = ranked_text.encode(text='class22:O').properties(title=alt.TitleParams(text='class2', align='right'))
row_number = ranked_text.encode(text='row_number:O').properties(title=alt.TitleParams(text='row_number', align='right'))


#print(pd.DataFrame.from_dict(count.to_dict(), orient='tight'))
text = alt.hconcat(#row_number,
                   layers,
                   feature,
                   count,
                   #class1,
                  # class2
                   ) # Combine data tables
for i in feature.to_dict():
    print(i)



bar = alt.Chart(source).mark_bar().encode(
    y=alt.Y('feature:O'), #, axis=alt.Axis(orient='left')
    x=alt.X('count(jmd):Q'),
).transform_filter(click
).properties(
    height=1500,
    width=100,
)

sql_filter33 = f"""
------------------------------------------------------------------
---  get values of feature where jmd > 1.3
------------------------------------------------------------------
SELECT
	uavfeaturestatistic.class_name,
	ROUND(uavfeaturestatistic.value, 2) as value,
	substr(class_name, 1, instr(class_name, '_')-1) crop_type_code, 
	substr(uavfeaturestatistic.class_name, instr(uavfeaturestatistic.class_name, '_')+1, 2) phenogroup, 
	jmd_table.feature, 
	jmd_table.class, 
	jmd_table.jmd, 
	COUNT(uavfeaturestatistic.class_name) as pixel_amount
FROM 
	uavfeaturestatistic
LEFT JOIN ( 
					SELECT
						uavfeaturejmd.feature, 
						uavfeaturejmd.class1 as class,
						--class2, 
						uavfeaturejmd.jmd
					FROM
						uavfeaturejmd
					----WHERE 
					--	uavfeaturejmd.jmd > 1.3
					--AND uavfeaturejmd.class1 LIKE 'SG%'

						
					UNION
					SELECT
						uavfeaturejmd.feature, 
						--class1,
						uavfeaturejmd.class2 as class, 
						uavfeaturejmd.jmd
					FROM
						uavfeaturejmd
					--WHERE 
						--uavfeaturejmd.jmd > 1.3
					--AND uavfeaturejmd.class1 LIKE 'SG%'
					ORDER BY 
						uavfeaturejmd.jmd DESC,
						uavfeaturejmd.class1, 
						uavfeaturejmd.class2,
						uavfeaturejmd.feature ) as jmd_table

ON (uavfeaturestatistic.class_name == jmd_table.class AND uavfeaturestatistic.feature == jmd_table.feature)
WHERE 
	(
	uavfeaturestatistic.class_name LIKE '{CROP_TYPE}%' 
	OR 
	uavfeaturestatistic.class_name LIKE 'UNK%'
	)
AND jmd_table.jmd NOT NULL
AND jmd_table.jmd > {JMD}
AND jmd_table.feature LIKE '{FEATURE[:-1]}%'

GROUP BY 
uavfeaturestatistic.class_name

ORDER BY
--	uavfeaturestatistic.refpanel,
--uavfeaturestatistic.class_name, 
--uavfeaturestatistic.feature, 
COUNT(uavfeaturestatistic.class_name)
;

"""
dt_stat = cfg.query_sql(sql_filter33, db=cfg.dbarchive3.archive.engine)
source = dutils.feature_layer_split(dt_stat)['df']


barplot = alt.Chart(source).mark_bar().encode(
            x='phenogroup:N',
            y='pixel_amount:Q',
            color='crop_type_code:N',
            #column='site:N'
#).encode(row='phenogroup'
).properties(width=1300, height=200).configure_axis(
    labelFontSize=16,
    titleFontSize=16
)

st.write(barplot)



sql_filter3 = f"""

------------------------------------------------------------------
---  get values of feature where jmd > 1.3
------------------------------------------------------------------
SELECT
	uavfeaturestatistic.class_name,
	ROUND(uavfeaturestatistic.value, 2) as value,
	--substr(class_name, 1, instr(class_name, '_')-1) crop_type_code, 
	substr(uavfeaturestatistic.class_name, instr(uavfeaturestatistic.class_name, '_')+1, 2) phenogroup, 
	jmd_table.feature, 
	jmd_table.class, 
	jmd_table.jmd
FROM 
	uavfeaturestatistic
LEFT JOIN ( 
					SELECT
						uavfeaturejmd.feature, 
						uavfeaturejmd.class1 as class,
						--class2, 
						uavfeaturejmd.jmd
					FROM
						uavfeaturejmd
					----WHERE 
					--	uavfeaturejmd.jmd > 1.3
					--AND uavfeaturejmd.class1 LIKE 'SG%'

						
					UNION
					SELECT
						uavfeaturejmd.feature, 
						--class1,
						uavfeaturejmd.class2 as class, 
						uavfeaturejmd.jmd
					FROM
						uavfeaturejmd
					--WHERE 
						--uavfeaturejmd.jmd > 1.3
					--AND uavfeaturejmd.class1 LIKE 'SG%'
					ORDER BY 
						uavfeaturejmd.jmd DESC,
						uavfeaturejmd.class1, 
						uavfeaturejmd.class2,
						uavfeaturejmd.feature ) as jmd_table

ON (uavfeaturestatistic.class_name == jmd_table.class AND uavfeaturestatistic.feature == jmd_table.feature)
WHERE 
	(
	uavfeaturestatistic.class_name LIKE '{CROP_TYPE}%' 
	--OR 
	--uavfeaturestatistic.class_name LIKE 'UNK%'
	)
AND jmd_table.jmd NOT NULL
AND jmd_table.jmd > {JMD}
--AND jmd_table.feature LIKE '{FEATURE[:-1]}%'


ORDER BY
--	uavfeaturestatistic.refpanel,
uavfeaturestatistic.class_name, 
uavfeaturestatistic.feature
;
"""

dt_phen = cfg.query_sql(sql_filter3, db=cfg.dbarchive3.archive.engine)
data = dutils.feature_layer_split(dt_phen)['df']
print(data.columns)
print(data.head())



box_plot2 = alt.Chart(data
#).transform_filter(
# 'datum.value != null'
#).transform_aggregate(
    #count='count()',
    #v_value = 'value',
#    avg_value='mean(value)',
#    std_value='stdev(value)',
#    groupby=['feature']
).transform_joinaggregate(
  # v_value='value',
   avg_value = 'mean(value)',
   std_value = 'stdev(value)',

   min_value = 'min(value)',
   max_value = 'max(value)',
   #groupby = ['feature', 'class_name']
    groupby = ['feature']

).transform_calculate(
    z_value="(datum.value - datum.avg_value) / datum.std_value",
    norm_value="(datum.value - datum.min_value) /(datum.max_value - datum.min_value)"

).mark_boxplot(size=30, extent=0.5, outliers={'size': 5}, opacity=0.5 # extent='min-max',.encode(
).encode(
    x=alt.X('layers:N', title=None, axis=alt.Axis(labels=True, ticks=True), scale=alt.Scale(padding=1), sort=None),
    y=alt.Y('norm_value:Q', scale=alt.Scale(domain=[-0.5, 1.5])),
   #y=alt.Y('norm_value:Q'),
    color=alt.Y('class_name:N', scale=alt.Scale(zero=False)),
    row = 'phenogroup'
    #column='phenogroup'
    #color=alt.condition(selector, 'class_name:N', alt.ColorValue('gray'))
#).add_selection(
#        brush
).properties(width=1300, height=200).configure_axis(
    labelFontSize=16,
    titleFontSize=16
)

st.write(box_plot2)




'''


sql_filter_old = f"""
SELECT 
	uavfeaturestatistic.class_name,
	ROUND(uavfeaturestatistic.value, 2) as value,
	--substr(class_name, 1, instr(class_name, '_')-1) crop_type_code, 
	substr(uavfeaturestatistic.class_name, instr(uavfeaturestatistic.class_name, '_')+1, 2) phenogroup, 
	uavfeaturestatistic.fid,
	uavfeaturestatistic.aoi,
	uavfeaturestatistic.refpanel,
	jmd_table.feature
FROM (
			SELECT * FROM (
			SELECT
				feature, 
				COUNT(jmd) as count_jmd

			FROM
				uavfeaturejmd

			WHERE 
				jmd >= {JMD}
			AND (class1  LIKE '{CROP_TYPE}%' OR class2 LIKE '{CROP_TYPE}%' OR class1 LIKE 'UNK%' OR class2 LIKE 'UNK%' )

			GROUP BY
				feature

			ORDER BY 
				COUNT(jmd) DESC,
				feature 
				) 
			WHERE count_jmd == MAX(count_jmd)
						) as jmd_table

LEFT JOIN 
uavfeaturestatistic
ON (uavfeaturestatistic.feature == jmd_table.feature)

WHERE 
	(uavfeaturestatistic.class_name LIKE '{CROP_TYPE}%' OR uavfeaturestatistic.class_name LIKE 'UNK%')
--AND jmd_table.jmd NOT NULL

AND (uavfeaturestatistic.class_name LIKE '{CROP_TYPE}%'  OR uavfeaturestatistic.class_name LIKE 'UNK%')

ORDER BY
--	uavfeaturestatistic.refpanel,
uavfeaturestatistic.class_name, 
uavfeaturestatistic.feature
;
"""
sql_filter2 = f"""

SELECT
	uavfeaturestatistic.class_name,
	uavfeaturestatistic.feature,
	ROUND(uavfeaturestatistic.value, 2) as value,
	--substr(class_name, 1, instr(class_name, '_')-1) crop_type_code, 
	substr(uavfeaturestatistic.class_name, instr(uavfeaturestatistic.class_name, '_')+1, 2) phenogroup, 

	uavfeaturestatistic.fid,
	uavfeaturestatistic.aoi,
	uavfeaturestatistic.bbch,
	uavfeaturestatistic.crop_type_code,
	uavfeaturestatistic.uav_date,
	--uavfeaturestatistic.refpanel,
	--jmd_table.feature,
	--COUNT(uavfeaturejmd.class_name),
	jmd_table.jmd,
	jmd_table.class,
	jmd_table.class_to_compare,
	jmd_table.refpanel
	--COUNT(jmd_table.feature) as count_feature

FROM 
	uavfeaturestatistic

LEFT JOIN 
		(
					SELECT
							uavfeaturejmd.feature, 
							uavfeaturejmd.jmd, 
							uavfeaturejmd.class1 as class,
							uavfeaturejmd.class2 as class_to_compare,
							uavfeaturejmd.refpanel
						FROM
							uavfeaturejmd
						--WHERE 
							--uavfeaturejmd.class1  LIKE 'WW%'
							--OR (uavfeaturejmd.class2 LIKE 'WW%' AND uavfeaturejmd.class1 LIKE 'UNK%')

						--UNION ALL

						--SELECT
						--	uavfeaturejmd.feature, 
						--	uavfeaturejmd.jmd, 
						--	uavfeaturejmd.class2 as class,
						--	uavfeaturejmd.class1 as class_to_compare,
						--	uavfeaturejmd.refpanel
						--FROM
						--	uavfeaturejmd
						----WHERE 
						--	--uavfeaturejmd.class2 LIKE 'WW%'
						--	--OR (uavfeaturejmd.class1 LIKE 'WW%' AND uavfeaturejmd.class2 LIKE 'UNK%')
							) as jmd_table

ON (uavfeaturestatistic.feature == jmd_table.feature AND uavfeaturestatistic.refpanel == jmd_table.refpanel AND uavfeaturestatistic.class_name == jmd_table.class)

WHERE 
	jmd_table.jmd IS NOT NULL
AND 
	uavfeaturestatistic.refpanel != 'noRP'
AND 
	jmd_table.jmd >= {JMD}
AND 
	uavfeaturestatistic.crop_type_code LIKE '{CROP_TYPE}%' 
	OR (uavfeaturestatistic.crop_type_code LIKE 'UNK%' AND jmd_table.class_to_compare  LIKE '{CROP_TYPE}%')

--GROUP BY 
	--jmd_table.feature, 
	--jmd_table.class, 
	--uavfeaturestatistic.crop_type_code

ORDER BY 
	uavfeaturestatistic.crop_type_code
;
"""

data = cfg.query_sql(sql_filter2)

print(data.head())
#selector = alt.selection(type='interval')  # selection of type "interval"
#selection = alt.selection_multi(fields=['class_name:N'], bind='legend')
#brush = alt.selection_interval(encodings=['x'])

box_plot = alt.Chart(data
#).transform_filter(
# 'datum.value != null'
#).transform_aggregate(
    #count='count()',
    #v_value = 'value',
#    avg_value='mean(value)',
#    std_value='stdev(value)',
#    groupby=['feature']
).transform_joinaggregate(
  # v_value='value',
   avg_value = 'mean(value)',
   std_value = 'stdev(value)',

   min_value = 'min(value)',
   max_value = 'max(value)',
   #groupby = ['feature', 'class_name']
    groupby = ['feature']

).transform_calculate(
    z_value="(datum.value - datum.avg_value) / datum.std_value",
    norm_value="(datum.value - datum.min_value) /(datum.max_value - datum.min_value)"

).mark_boxplot(size=30, extent=0.5, outliers={'size': 5}, opacity=0.5 # extent='min-max',.encode(
).encode(
    x=alt.X('feature:N', title=None, axis=alt.Axis(labels=True, ticks=True), scale=alt.Scale(padding=1)),
    y=alt.Y('z_value:Q', scale=alt.Scale(domain=[-25, 25])),
   #y=alt.Y('norm_value:Q'),
    color=alt.Y('class_name:N', scale=alt.Scale(zero=False)),
    #row = 'phenogroup'
    column='feature'
    #color=alt.condition(selector, 'class_name:N', alt.ColorValue('gray'))
#).add_selection(
#        brush
).properties(width=1500, height=800).configure_axis(
    labelFontSize=16,
    titleFontSize=16
)



st.write(box_plot)
'''

#st.write('<style>div.block-container{padding-top:7rem;}</style>', unsafe_allow_html=True)
