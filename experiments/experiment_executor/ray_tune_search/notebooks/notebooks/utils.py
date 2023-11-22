import pandas as pd
import yaml
import matplotlib.pyplot as plt
import pandas as pd

def read_csv(dataset,type_simclr,n_transformations,percentage,dir):
    try:
        if(n_transformations==-1):
            folder=f'../{dir}/{dataset}/'
            folder_percentage=f'{type_simclr}_{dataset}_P{percentage}/data.csv'
        else:
            folder=f'../{dir}/{type_simclr}/{dataset}/'
            folder_percentage=f'{type_simclr}_{n_transformations}T_{dataset}_P{percentage}/data.csv'
            
        
        data = pd.read_csv(f'{folder}{folder_percentage}')
        data.loc[data['score'] < 0,'score'] = 0
    except:
        print(f'{folder}{folder_percentage}')
        data = pd.DataFrame({'Unnamed: 0': [], 'score': []})
    return data

def get_no_reducer_score(dataset):
    with open(f'TV_sb_no_reducer/scores/no_reducer_{dataset}.yaml') as f:
        score = yaml.load(f, Loader=yaml.FullLoader)
        score = score['score']
        #print(score)
    return score


def fixed_percentage_models(
        datasets=['kuhar', 'motionsense', 'uci', 'wisdm', 'realworld_thigh', 'realworld_waist'],
        models=['simclr_linear','simclr','simclr_full'],
        markers = ['d', 's', 'p', 'h', 'o'],
        percentages=[25, 50, 75, 100, 200],
        colors = ['blue', 'orange', 'lightgreen', 'darkgreen', 'purple'],
        y_lim = [0, 1],
        n_transformations=-1,
        dir="experiments/simclr_all/all_transformations"
        ):
    """
    Plots the best accuracy for each model and dataset for a fixed percentage of the dataset.
    """
    
    for dataset in datasets:
        with open(f'../TV_sb_no_reducer/scores/no_reducer_{dataset}.yaml') as f:
            no_reducer_score = yaml.load(f, Loader=yaml.FullLoader)
            no_reducer_score = no_reducer_score['score']
        
        for i, percentage in enumerate(percentages):
            # Subplots for each dataset
            fig, axs = plt.subplots(1, 2, figsize=(15, 8))
            plt.suptitle(f'Up to {percentage}% on {dataset}')
            max_length = 0
            for j, model in enumerate(models):
                dir1 =f'{dir}/{model}/best_transf'
                data = read_csv(dataset, model, n_transformations, percentage,dir1)
                try:
                    data = read_csv(dataset, model, n_transformations, percentage,dir1)


                
                except:
                    data = pd.DataFrame({'Unnamed: 0': [], 'score': []})
                max_index = data['Unnamed: 0'].max()
                if  max_index > max_length:
                    max_length = max_index
                axs[0].plot(data['Unnamed: 0'], data['score'], markers[j], color=colors[j], markersize=4, label=model)
                axs[1].plot(data['score'].cummax(), color=colors[j], linewidth=2, label=model)
            axs[0].plot([0, max_length], [no_reducer_score, no_reducer_score], color='red', linewidth=2, label='No reducer')
            axs[0].legend(loc='lower center')
            axs[0].set_ylabel(f'Best accuracy')
            axs[0].set_xlabel('Iterations')
            axs[0].grid()
            axs[0].set_ylim(y_lim)
            axs[0].set_facecolor('#e6f5c9')
            axs[1].plot([0, max_length], [no_reducer_score, no_reducer_score], color='red', linewidth=2, label='No reducer')
            axs[1].legend(loc='lower center')
            axs[1].set_ylabel(f'Best accuracy')
            axs[1].set_xlabel('Iterations')
            axs[1].grid()
            axs[1].set_ylim(y_lim)
            axs[1].set_facecolor('#e6f5c9')

            plt.show()


def fixed_percentage_transf(
        datasets=['kuhar', 'motionsense', 'uci', 'wisdm', 'realworld_thigh', 'realworld_waist'],
        model='simcrl_full',
        markers = ['d', 's', 'p', 'h', 'o'],
        percentages=[25, 50, 75, 100, 200],
        colors = ['blue', 'orange', 'lightgreen', 'darkgreen', 'purple'],
        y_lim = [0, 1],
        n_transformations_array=[1,2,3,4],
         dir="experiments/simclr_all/by_n_transformations"
        ):
    """
    Plots the best accuracy for each model and dataset for a fixed percentage of the dataset.
    """
    for dataset in datasets:
        with open(f'../TV_sb_no_reducer/scores/no_reducer_{dataset}.yaml') as f:
            no_reducer_score = yaml.load(f, Loader=yaml.FullLoader)
            no_reducer_score = no_reducer_score['score']
        
        for i, percentage in enumerate(percentages):
            # Subplots for each dataset
            fig, axs = plt.subplots(1, 2, figsize=(15, 8))
            plt.suptitle(f'Up to {percentage}% on {dataset}')
            max_length = 0
            for j, n_transformations in enumerate(n_transformations_array):
                data = read_csv(dataset, model, n_transformations, percentage,dir)
                try:
                    data = read_csv(dataset, model, n_transformations, percentage,dir)


                
                except:
                    data = pd.DataFrame({'Unnamed: 0': [], 'score': []})
                max_index = data['Unnamed: 0'].max()
                if  max_index > max_length:
                    max_length = max_index
                axs[0].plot(data['Unnamed: 0'], data['score'], markers[j], color=colors[j], markersize=4, label=f'{n_transformations}Transf')
                axs[1].plot(data['score'].cummax(), color=colors[j], linewidth=2, label=f'{n_transformations}Transf')
            axs[0].plot([0, max_length], [no_reducer_score, no_reducer_score], color='red', linewidth=2, label='No reducer')
            axs[0].legend(loc='lower center')
            axs[0].set_ylabel(f'Best accuracy')
            axs[0].set_xlabel('Iterations')
            axs[0].grid()
            axs[0].set_ylim(y_lim)
            axs[0].set_facecolor('#e6f5c9')
            axs[1].plot([0, max_length], [no_reducer_score, no_reducer_score], color='red', linewidth=2, label='No reducer')
            axs[1].legend(loc='lower center')
            axs[1].set_ylabel(f'Best accuracy')
            axs[1].set_xlabel('Iterations')
            axs[1].grid()
            axs[1].set_ylim(y_lim)
            axs[1].set_facecolor('#e6f5c9')

            plt.show()