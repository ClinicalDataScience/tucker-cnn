import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from totalsegmentator.map_to_binary import class_map, class_map_5_parts
from copy import copy

def aggregate(df):
    filtered_df = df[(df['ds'] != -1) & (df['nsd'] != -1)]
    means = filtered_df.groupby('label_str')[['ds', 'nsd']].mean()
    return means


def plot_diff(ref_dict, pred_dict, names=["baseline", "prediction"], entity="dice", title=None, save=None):
    ref_dict, pred_dict = ref_dict[entity], pred_dict[entity]
    # Get all unique labels from v2
    all_labels = list(ref_dict.keys())
    
    v1_values = [pred_dict[k] for k in all_labels]
    v2_values = [ref_dict[k] for k in all_labels]
    # Calculate differences
    differences = [v1 - v2 for v1, v2 in zip(v1_values, v2_values)]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(20, 8))  # Adjust the figure size as needed
    bar_width = 0.35
    index = range(len(all_labels))
    
    bars = ax.bar(index, differences, bar_width)
    
    ax.axhline(0, color='black', linewidth=1)  # Center line at zero
    ax.axhline(np.mean(differences), color='red', linewidth=1)  # Prediction mean offset 
    
    ax.set_xlabel('Labels')
    ax.set_ylabel(f'Difference ({names[1]} - {names[0]}) in {entity}')
    if title is None:
        ax.set_title(f'Difference Chart between {names[1]} and {names[0]} Values')
    else:
        ax.set_title(title)
    ax.set_xticks([i for i in index])
    ax.set_xticklabels(all_labels, rotation=90)
    
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_hist_by_entity(pred_dict_1, entity, file_path_template, diff=True, save=None):
    def aggregate(df):
        filtered_df = df[(df['ds'] != -1) & (df['nsd'] != -1)]
        means = filtered_df.groupby('label_str')[['ds', 'nsd']].mean()
        return means
    
    def process_data(pred_dict_1, entity, class_map=None, diff=diff):
        all_differences = []
        
        for i in range(9):
            ref = copy(pred_dict_1)
            
            # Read CSV file
            df = pd.read_csv(file_path_template.format(i))
            #df = pd.read_csv(f'/data/core-rad/data/tucker_results/results_grid_fine/0{i}.csv')
            means = aggregate(df)
            pred_dict_n = {"dice": means["ds"].to_dict(), "nsd": means["nsd"].to_dict()}
            
            ref, pred_dict_n = ref[entity], pred_dict_n[entity]
            
            # Determine labels based on the given class map
            if class_map:
                all_labels = list(class_map_5_parts[class_map].values())
            else:
                all_labels = list(ref.keys())

            v1_values = [pred_dict_n[k] for k in all_labels]
            v2_values = [ref[k] for k in all_labels]

            if diff:
                # Calculate differences
                differences = [v1 - v2 for v1, v2 in zip(v1_values, v2_values)]
            else:
                differences = v1_values
            all_differences.append(differences)
            
        return all_differences
    
    # Process original data
    all_differences_total = process_data(pred_dict_1, entity)
    
    # Process data for each map
    class_maps = [
        "class_map_part_organs",
        "class_map_part_vertebrae",
        "class_map_part_cardiac", 
        "class_map_part_muscles",
        "class_map_part_ribs"
    ]
    
    all_differences_maps = []
    for class_map in class_maps:
        all_differences_map = process_data(pred_dict_1, entity, class_map=class_map)
        all_differences_maps.append(all_differences_map)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = np.arange(9) * 1.5 + 1  # Increase the distance between each step
    width = 0.15
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Define colors for each model
    
    
    
    # Before plotting
    handles = []
    labels = []
    
    # Inside the loop where you create boxplots
    for i, position in enumerate(positions):
        for j, model_data in enumerate([all_differences_total[i]] + [differences[i] for differences in all_differences_maps]):
            x = position + (j - 2) * width
            box = ax.boxplot(model_data, positions=[x], widths=width, showfliers=False, patch_artist=True,
                             boxprops=dict(facecolor=colors[j]))  # Assign color to each model
            if j == 0:  # Only collect the boxplot for total model for the legend
                handles.append(box['boxes'][0])
                labels.append('Total')
            else:
                handles.append(box['boxes'][0])
                labels.append(class_maps[j-1].split("_")[-1])
    
    # After plotting
    ax.legend(handles[:6], labels[:6], loc='upper right')
    names = [1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    # Set labels and title
    ax.set_xlabel('Compression factor')
    ax.set_ylabel('Dice')
    if diff:
        ax.set_ylabel('Dice Differences')
    ax.set_title('Whisker Plot of Dice Differences for Different Models at Each Step')
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in names])#[f'Step {i+1}' for i in range(9)])
    #ax.legend(['Total'] + ['organs'] + ['vertebrae'] + ['cardiac'] + ['muscles'] + ['ribs'], loc='upper right')
    plt.grid(True)

    if save is not None:
        plt.savefig(save)
    plt.show()


def get_means(pred_dict_1, entity, file_path_template, class_map=None):
    all_means = []
    all_means2 = []
    all_stds = []
    std_diff = []
    
    for i in range(9):
        
        ref = copy(pred_dict_1)
        
        df = pd.read_csv(file_path_template.format(i))
        
        means = aggregate(df)
        pred_dict_n = {"dice": means["ds"].to_dict(), "nsd": means["nsd"].to_dict()}
        
        ref, pred_dict_n = ref[entity], pred_dict_n[entity]
        # Determine labels based on the given class map
        if class_map:
            all_labels = list(class_map_5_parts[class_map].values())
        else:
            all_labels = list(ref.keys())
        
        v1_values = [pred_dict_n[k] for k in all_labels]
        v2_values = [ref[k] for k in all_labels]
        
        # Calculate differences
        differences = [v1 - v2 for v1, v2 in zip(v1_values, v2_values)]
        all_means.append(np.mean(differences))
        all_means2.append(np.mean(v1_values))
        std_diff.append(np.std(differences))
        
    return all_means, std_diff
