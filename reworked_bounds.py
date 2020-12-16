import matplotlib.pyplot as plt 
import pandas
import numpy as np
import pulp

## This code is an edit of Alexander Jaffe's metagame bounds script bounds.py available: https://github.com/blinkity/metagame
## My comments are marked with double ##s Jaffe's original comments are marked with a single #
## Removed some imports that seemed to be unnecessary
## Deleted the boundsFromCsv function as I dont have the file and allData is never used
## Removed makeMatchups function as it references files I do not have access to and the code still functioned without it
## Removed makeMatchupsFromOverallBeatProbs for the same reason


## added the pulp. prefix to lpProblem, lpVariable and lpSum in setupBasicProblem below. The rest is the same as original

def setupBasicProblem(matrix):
    prob = pulp.LpProblem("rock_paper_scissors", pulp.LpMaximize)
    the_vars = np.append(matrix.index.values, (["w"]))
    lp_vars = pulp.LpVariable.dicts("vrs", the_vars)
#First add the objective function.
    prob += pulp.lpSum([lp_vars['w']])  
#Now add the non-negativity constraints.
    for row_strat in matrix.index.values:
        prob += pulp.lpSum([1.0 * lp_vars[row_strat]]) >= 0
#Now add the sum=1 constraint.
    prob += pulp.lpSum([1.0 * lp_vars[x] for x in matrix.index.values]) == 1
#Now add the column payoff constraints
    for col_strat in matrix.columns.values:
        stratTerms = [matrix.loc[row_strat, col_strat] * lp_vars[row_strat] for row_strat in matrix.index.values]
        allTerms = stratTerms + [-1 * lp_vars['w']]
        prob += pulp.lpSum(allTerms) >= 0
#now write it out and solve
    return prob, lp_vars

## This function is unedited
def solveGame(matrix):
    prob, lp_vars = setupBasicProblem(matrix)
    prob.writeLP("rockpaperscissors.lp")
    prob.solve()
#now prepare the value and mixed strategy
    game_val = pulp.value(lp_vars['w'])
    strat_probs = {}
    for row_strat in matrix.index.values:
        strat_probs[row_strat] = value(lp_vars[row_strat])
#and output it
    return prob, game_val, strat_probs

## This function is unedited
def solveGameWithRowConstraint(matrix, rowname, constraint):
    prob, lp_vars = setupBasicProblem(matrix)
#add the additional constraint
    prob += pulp.lpSum(lp_vars[rowname]) == constraint
    prob.writeLP("rockpaperscissors.lp")
    prob.solve()
#now prepare the value and mixed strategy
    game_val = pulp.value(lp_vars['w'])
    strat_probs = {}
    for row_strat in matrix.index.values:
        strat_probs[row_strat] = pulp.value(lp_vars[row_strat])
#and output it
    return prob, game_val, strat_probs

## This function is unedited

def getWinRates(rowname,matrix,division=10):
    probs = np.linspace(0,1,division+1)
    return pandas.Series([solveGameWithRowConstraint(matrix, rowname, p)[1] for p in probs], index=probs, name=rowname)

## This function is unedited
def getAllWinRates(matrix,division=10):
    return pandas.concat([getWinRates(row,matrix,division) for row in matrix.index.values], axis=1)   

## Removed the if/else statements as I do not need them
## Edited line 6 for better spacing on graphs
def plotIntervals(winRates,doSort,threshold):
    intervals = winRates.apply(lambda x: pandas.Series([x[x >= threshold].first_valid_index(), x[x >= threshold].last_valid_index()], index = ['minv','maxv'])).T
    intervals['bar1'] = intervals['minv']
    intervals['bar2'] = intervals['maxv'] - intervals['minv']
    intervals['bar3'] = 1 - (intervals['bar1'] + intervals['bar2'])
    img = intervals[['bar1','bar2','bar3']].plot(kind='barh',stacked=True, color=['w','g','w'], xticks = np.linspace(0,1, num=11, endpoint=True), legend=False)
    return img


## changed matchups to df below

def main():
            #insert filepath for input data
            df = pandas.read_csv("C:\\Users\\Rapid\\Desktop\\Metagame Bounds\\metagame-master\\matchups.csv", header=None, index_col = 0) ##filepath for matplotlib
            matchupPayoffs = 2*df - 1 
            allWinRates = getAllWinRates(matchupPayoffs,10)
            img = plotIntervals(allWinRates,True,-0.02)
            #insert filepath for output image
            plt.savefig("C:\\Users\\Rapid\\Desktop\\Des 405\\Portfolio\\Metagame Bounds\\matchup graph.png", bbox_inches='tight') ##img.get_figure() was not working so using this method to save the image as png instead. The file path is where the image is saved, name of the image and the file type to save as (.png)


if __name__ == "__main__":
    main()
