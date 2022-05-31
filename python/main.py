from utils.model_enums import model_types
from unityagents import UnityEnvironment
import numpy as np
from numpy import loadtxt

import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
import os
from collections import deque

import maddpg
import mad4pg

def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper left')
    ax.set(xlabel='# games', ylabel='score')
    plt.show(block=True)

def watchSmartAgent(env, config, dqnType):
    if dqnType == model_types.maddpg:
        maddpg.watchAgent(config, env)
    elif dqnType == model_types.mad4pg:
        mad4pg.watchAgent(config, env)
    # elif dqnType == model_types.mappo:
    #     maddpg.watchAgent(env, config)

def saveScores(filename, scores):
    dirName = 'scores'
    dirExists = os.path.isdir(dirName)
    if not dirExists:
        os.mkdir(dirName) 

    filename = os.path.join(dirName, filename) + '.txt'
    with open(filename, "w") as f:
        for s in scores:            
            f.write(str(s) +"\n")

    print('\nScores Saved to {:s}'.format(filename))


def run(isTest, dqnType):
    show_Graphics = isTest
    # please do not modify the line below
    env = UnityEnvironment(file_name="../p3_collab-compet/Tennis_Windows_x86_64/Tennis.exe", no_graphics = show_Graphics)
    config = Config()

    config.model = dqnType
    if isTest:
        if dqnType == model_types.maddpg:
            scores, scores_window = maddpg.Run(env, config)
        elif dqnType == model_types.mad4pg:
            scores, scores_window = mad4pg.Run(env, config)

        per = ""
        if config.USE_PRIORITY_REPLAY == True:
            per = "_per"
        filename = str(dqnType.name) + per
        saveScores(filename, scores)       

        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(len(scores), np.mean(scores_window)))
        else:
            print('\nEnvironment failed to solve in {:d} episodes.\tAverage Score: {:.2f}'.format(len(scores), np.mean(scores_window)))
        
        plot_seaborn(np.arange(len(scores)), scores, True)

    else:
        watchSmartAgent(env, config, dqnType)


def moving_average(a, n=3) :
    scores_window = deque(maxlen=n)  # last 100 scores
    scores_average = []
    for score in a:
        scores_window.append(score)
        scores_average.append(np.mean(scores_window))
    return scores_average 


def showScores():
    plt.figure(figsize=(13,8))   

    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")

    #Iterate through each file in saved scores
    for filename in os.listdir('scores'):
        file = os.path.join('scores', filename)
        # checking if it is a file
        if os.path.isfile(file):
            print(file)
            if not file.endswith('.txt'):
                continue
            scores = loadtxt(file, comments="#", delimiter=",", unpack=False)     
            iterations = np.arange(len(scores))
            target_score = [.5]*len(scores)
            
            # ax = sns.lineplot(
            #         np.array([iterations])[0],
            #         np.array([scores])[0],
            #         label=os.path.basename(file).split('.')[0],                    
            #     )

            scores = moving_average(scores,100)

            avgText = os.path.basename(file).split('.')[0] + '-avg'

            solvedIteration = next(x[0] for x in enumerate(scores) if x[1] > 0.5)
            print(avgText + "   Environment solved in:" + str(solvedIteration) )

            ax2 = sns.lineplot(
                    np.array([iterations])[0],
                    np.array([scores])[0],
                    label = avgText,
                    linestyle='--',
                )  
            
            ax3 = sns.lineplot(
                    np.array([iterations])[0],
                    np.array([target_score])[0],
                    color='black',
                    linestyle=':'          
                )
            ax3.legend(loc='lower right')
            ax3.set(xlabel='# games', ylabel='score') 
            
    plt.show(block=True)

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed 
    #run(isTest = True, dqnType= model_types.maddpg)

    #show all scores in scores folder
    showScores()