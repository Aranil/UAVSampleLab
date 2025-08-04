"""
To plot the accuracy Metrics - All Crop type models in one Plot
Reads Data from DB table -> 'uavcnnaccuracyreport'
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import numpy as np

import config as cfg
from dbflow.src import db_utility as dbu
from accuracy_assessment import accuracy_utils as acc




# Set font to Arial
plt.rcParams['font.family'] = 'Arial'


def draw_heatmap(data, metric, ax, cmap='RdYlGn', _cbar=False, subplot_number=None, cbar_val_range=[0,1], _fmt=".2f",
                 x_axis_labels='auto', y_axis_labels='auto', _annot_kws={'fontsize': 16}, _fontsize=24, number_sublots=3):
    """
    function to plot one heatmap

    Author: https://github.com/friedmag-m

    Parameters
    ----------
    data: pandas dataframe
    metric: subplots in horizontal direction
    ax: matplotlib axis
    cmap: matplotlib cmap
    _cbar: boolean
        if True cbar is plotted for each subplot
    subplot_number: if None - all labels on y-axis are plotted, if increment (as int)  - number of subplots [0,1,2,3..]
    passed through only for a first plot with i=0 labels are plotted

    Returns
    -------

    """
    sns.heatmap(data=data,
                ax=ax,
                annot=True,
                fmt=_fmt,
                cbar=_cbar,
                square=True,
                cmap=cmap,
                vmin=cbar_val_range[0], vmax=cbar_val_range[1],
                annot_kws=_annot_kws,
                xticklabels=x_axis_labels if i != number_sublots else False,
                yticklabels=y_axis_labels,
                )

    ax.set_title(metric, fontsize=_fontsize, pad=20)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(' ', fontsize=18)
    ax.set_ylabel(' ', fontsize=18, rotation=90)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
   # ax.set_yticklabels(ax.get_yticklabels(), rotation=90, ha='right')


    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    #https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

    #if metric != 'f1':
    if subplot_number != None:
        if subplot_number != 0:
            ax.set_yticklabels([])  # Empty y-Tick-Labels for maps other than 'f1'
            ax.set_ylabel(' ')

    #cbar = ax.collections[0].colorbar
    #cbar.ax.tick_params(labelsize=14)
    return ax




if __name__ == '__main__':


    # connect to DB
    dbarchive = dbu.connect2db(cfg.db_path)


    PLOT_SET = 'metrics'
    #PLOT_SET = 'SampleNumber'

    # input variables
    UNIT = ['reflectance', 'DN']

    SAMPLE_USED_IN_CNN_TRAINING = 'no'
    #SAMPLE_USED_IN_CNN_TRAINING = 'yes'

    y_row_number = 3
    #y_row_number = 4

    # define input variables
    if PLOT_SET == 'metrics':
        metrics = [

            #'sensitivity',
            #'specificity',
            'f1',
            'precision',
            'accuracy',
            #'precision',
            #'recall'
            #' '
            # 'false_negative_rate',
            # 'false_positive_rate',
            # 'true_negative_rate',
            # 'true_positive_rate'
        ]


    elif PLOT_SET == 'SampleNumber':
        metrics = [
                    #'number_of_TP_samples',
                    #'number_of_TN_samples',
                    #' ',
                    'number_of_FP_samples',
                    'number_of_FN_samples',
                    ' '
                    #'number_of_Total_samples'
                    ]

        title_labels = {
                    'number_of_TP_samples': 'Number of True Positive',
                    'number_of_TN_samples': 'Number of True Negative',
                    'number_of_FP_samples': 'Number of False Positive',
                    'number_of_FN_samples': 'Number of False Negative',
                    'number_of_Total_samples': 'Total Number of Samples',
                }



    outdir = pathlib.Path(cfg.output_path).joinpath(r'_plots\_accuracy')
    outdir.mkdir(parents=True, exist_ok=True)
    OUTFILE = outdir.joinpath("accuracy_all_{}_{}_{}_{}.png".format(UNIT, SAMPLE_USED_IN_CNN_TRAINING, PLOT_SET, metrics))



    # query table with metrics from db
    sql = f"""
            SELECT 
                metric, 
                class, 
                ROUND(value, 2) as value, 
                crop_type_code, 
                pattern, 
                unit, 
                sample_used_in_cnn
            FROM
                uavcnnaccuracyreport
            WHERE pattern=='xx'
                
            """

    sql_query = pd.read_sql(sql, dbarchive.archive.engine)
    df = pd.DataFrame(sql_query)


    #--------  select unit & if sample was used for training of CNN model
    filtered_df = df[
                    # (df['unit'] == UNIT)
                    # (df['crop_type_code'] == 'KM') &
                     (df['sample_used_in_cnn'] == SAMPLE_USED_IN_CNN_TRAINING) &
                     (df['class']).isin(cfg.classes) &
                     (df['metric'].isin(metrics))
                    ].reset_index()

    #print(filtered_df['metric'].unique())

    if PLOT_SET == 'SampleNumber': # change the headers of the plots
        filtered_df['metric'] = filtered_df['metric'].replace(title_labels)
        metrics = filtered_df['metric'].unique().tolist() + [' ']


    #--------  define sublot parameters
    num_plots = len(metrics)
    number_of_polots = len(metrics)

    #--------  Create a subplot grid
    #fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, y_row_number * 3))

    for i, metric in enumerate(metrics):

        # make data selection for one heatmap
        data_ = filtered_df[filtered_df['metric'] == metric]

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):

            data = data_.reset_index(drop=True).drop(columns=['index'])
            #data = data.set_index(['crop_type_code', 'unit'])
            #print(data)
            #duplicates = data.index.duplicated()
            #print(data[duplicates])

            #data = data.pivot_table(index=['crop_type_code', 'unit'], columns='class', values='value')#.reset_index()
            data = data.pivot_table(index=['crop_type_code', 'unit'], columns='class', values='value')  # .reset_index()
            print(data.head())



        #data.columns = [acc.tick_labels.get(col, col) for col in data.columns]  # Replace class names with tick labels
        #data.index = [cfg.crop_types.get(pattern, pattern) for pattern in data.index]  # Replace pattern names with mapping
        x_axis_labels = [acc.tick_labels.get(col, col) for col in data.columns]


        # use this to rewrite y-axis with multi index or index
        new_index_list = []
        for pattern in data.index:
            if isinstance(pattern, str): # for one index
                new_index_list.append(cfg.crop_types.get(pattern, pattern))
            elif isinstance(pattern, tuple): # for multi-index, crop_type_code must be the first value !
                pattern_list = list(pattern)
                for key, j in enumerate(pattern_list):
                    if key == 0:
                        j_main = cfg.crop_types.get(j, pattern) # j_main is for 'crop_type_code' column
                    #else:
                    #    print('Define new dictionary for y-axis pattern! ')
                    new_pattern = ['[' + k + ']' for k in pattern_list[1:]]
                    j_label = '\n'.join([j_main] + new_pattern)  # add res of the multi-indices to the main one

                    #j_label = '\n'.join([j_main] + pattern_list[1:]) # add res of the multi-indices to the main one


                new_index_list.append(j_label)


        # set new indices on y-axis
        data.index = new_index_list


        for s in metrics:
            if s == " ":
                number_sublots_ = number_of_polots -1
            else:
                number_sublots_ = number_of_polots


                # function to plot one heatmap in subplot
        if PLOT_SET == 'metrics':
            draw_heatmap(data, metric, axes[i],
                         x_axis_labels=x_axis_labels,
                         number_sublots=number_sublots_,
                         cmap='RdYlGn',
                         _cbar=False,
                         subplot_number=i,
                         _annot_kws={'fontsize': 16},
                         _fontsize=24)
        elif PLOT_SET == 'SampleNumber':            # ----use this for Number_of_Samples !!!
            draw_heatmap(data, metric, axes[i],
                         x_axis_labels=x_axis_labels,
                         number_sublots=number_sublots_,
                         cmap='YlOrBr', _cbar=False,
                         subplot_number=i,
                         cbar_val_range=[0, 300],
                         _fmt=".0f",
                         _annot_kws={'fontsize': 12},
                         _fontsize=24)


    #-------- Add colorbar
    # Create a separate axis for the color bar
   # if num_plots == 4:
   # cbar_ax = fig.add_axes([0.93, 0.32, 0.02, 0.36])  # [left, bottom, width, height]
    if number_sublots_ == 3:
        #cbar_ax = fig.add_axes([0.72, 0.315, 0.015, 0.32])  # [left, bottom, width, height]

        if y_row_number == 3:
            cbar_ax = fig.add_axes([0.93, 0.365, 0.015, 0.075 * y_row_number])  # [left, bottom, width, height]

        elif y_row_number == 4:
            cbar_ax = fig.add_axes([0.93, 0.375, 0.015, 0.21])  # [left, bottom, width, height]

        #elif num_plots == 3:
        #cbar_ax = fig.add_axes([0.655, 0.32, 0.02, 0.36])  # for two plots
        # Add the color bar to the figure
        cbar = fig.colorbar(axes[-1].collections[0], cax=cbar_ax)


    if number_sublots_ == 2:
        if y_row_number == 3:
            cbar_ax = fig.add_axes([0.655, 0.365, 0.015, 0.075 * y_row_number])  # [left, bottom, width, height]

        elif y_row_number == 4:
            cbar_ax = fig.add_axes([0.655, 0.375, 0.015, 0.21])  # [left, bottom, width, height]

        #elif num_plots == 3:
        #cbar_ax = fig.add_axes([0.655, 0.32, 0.02, 0.36])  # for two plots
        # Add the color bar to the figure
        cbar = fig.colorbar(axes[-1].collections[0], cax=cbar_ax)


    # Increase the label size of the colorbar
    cbar.ax.tick_params(labelsize=18)  # Adjust the fontsize here

    #plt.tight_layout()
    #plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.35)
    plt.savefig(OUTFILE.as_posix(), dpi=500)
    plt.show()
    plt.close()