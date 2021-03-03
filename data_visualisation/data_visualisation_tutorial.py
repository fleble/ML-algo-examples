#!/usr/bin/python3


######################## DATA VISUALISATION TUTORIAL ########################


# Load libraries
import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from plotnine import *

univariate_plotting   = 0
bivariate_plotting    = 1
box_violin_plotting   = 0
multivariate_plotting = 0
time_series           = 0

# Set matplotlib aesthetics
plt.rcParams.update(plt.rcParamsDefault)
#plt.style.use('classic')
plt.rcParams['patch.edgecolor'] = 'white'
plt.rcParams['patch.linewidth'] = 1
plt.rcParams['patch.facecolor'] = 'C0'
plt.rcParams['lines.markersize'] = 6
#plt.rcParams['lines.markerfacecolor'] = 'C0'
#plt.rcParams['lines.markeredgecolor']  = 'white'
#plt.rcParams['lines.markeredgewidth']  = 1.0 

# Figure
plt.rcParams['figure.figsize'] = 3, 2
#plt.rcParams['axes.xmargin'] = 0
#plt.rcParams['axes.ymargin'] = 0

# x ticks
plt.rcParams['xtick.top']            = False  ## draw ticks on the top side
plt.rcParams['xtick.bottom']         = True   ## draw ticks on the bottom side
plt.rcParams['xtick.labeltop']       = False  ## draw label on the top
plt.rcParams['xtick.labelbottom']    = True   ## draw label on the bottom
plt.rcParams['xtick.major.size']     = 3.5    ## major tick size in points
plt.rcParams['xtick.minor.size']     = 2      ## minor tick size in points
plt.rcParams['xtick.major.width']    = 0.8    ## major tick width in points
plt.rcParams['xtick.minor.width']    = 0.6    ## minor tick width in points
plt.rcParams['xtick.major.pad']      = 3.5    ## distance to major tick label in points
plt.rcParams['xtick.minor.pad']      = 3.4    ## distance to the minor tick label in points
plt.rcParams['xtick.color']          = 'black'  ## color of the tick labels
plt.rcParams['xtick.labelsize']      = 'small' ## fontsize of the tick labels
plt.rcParams['xtick.direction']      = 'out'    ## direction: in, out, or inout
plt.rcParams['xtick.minor.visible']  = False  ## visibility of minor ticks on x-axis
plt.rcParams['xtick.major.top']      = True   ## draw x axis top major ticks
plt.rcParams['xtick.major.bottom']   = True   ## draw x axis bottom major ticks
plt.rcParams['xtick.minor.top']      = True   ## draw x axis top minor ticks
plt.rcParams['xtick.minor.bottom']   = True   ## draw x axis bottom minor ticks
plt.rcParams['xtick.alignment']      = 'center' ## alignment of xticks

# y ticks
plt.rcParams['ytick.right']          = False  ## draw ticks on the top side
plt.rcParams['ytick.left']           = True   ## draw ticks on the bottom side
plt.rcParams['ytick.labelright']     = False  ## draw label on the top
plt.rcParams['ytick.labelleft']      = True   ## draw label on the bottom
plt.rcParams['ytick.major.size']     = 3.5    ## major tick size in points
plt.rcParams['ytick.minor.size']     = 2      ## minor tick size in points
plt.rcParams['ytick.major.width']    = 0.8    ## major tick width in points
plt.rcParams['ytick.minor.width']    = 0.6    ## minor tick width in points
plt.rcParams['ytick.major.pad']      = 3.5    ## distance to major tick label in points
plt.rcParams['ytick.minor.pad']      = 3.4    ## distance to the minor tick label in points
plt.rcParams['ytick.color']          = 'black'  ## color of the tick labels
plt.rcParams['ytick.labelsize']      = 'small' ## fontsize of the tick labels
plt.rcParams['ytick.direction']      = 'out'    ## direction: in, out, or inout
plt.rcParams['ytick.minor.visible']  = False  ## visibility of minor ticks on x-axis
plt.rcParams['ytick.major.right']    = True   ## draw x axis top major ticks
plt.rcParams['ytick.major.left']     = True   ## draw x axis bottom major ticks
plt.rcParams['ytick.minor.right']    = True   ## draw x axis top minor ticks
plt.rcParams['ytick.minor.left']     = True   ## draw x axis bottom minor ticks
plt.rcParams['ytick.alignment']      = 'center' ## alignment of xticks

# line
plt.rcParams['lines.linewidth'] = 1.2

#plt.rcParams['figure.facecolor'] = 'grey'
#plt.rcParams['scatter.edgecolor'] = 'white'

#plt.rcParams['patch.edgecolor'] = 'black'


#sns.set()
sns.set_style(rc={'patch.edgecolor': 'w','patch.force_edgecolor': True})

# Path of the files to read
pokemon_file_path = "./input/pokemon.csv"
wine_file_path = "./input/winemag-data_first150k.csv"
shelter_file_path = "./input/aac_shelter_outcomes.csv"

# Define datasets
df_pokemon = pd.read_csv(pokemon_file_path)
df_wine = pd.read_csv(wine_file_path)
df_wine_top5 = df_wine[df_wine.variety.isin(df_wine.variety.value_counts().head(5).index)]
df_shelter = pd.read_csv(shelter_file_path, parse_dates=['date_of_birth', 'datetime'])

# normal distribution 1D
normal_dist = np.random.normal(size=1000)
normal_dist2 = np.random.normal(size=200)

# normal distribution 2D
normal_dist_2D_n = 10000;
normal_dist_2D_n_sample = 300;
normal_dist_2D = np.random.multivariate_normal([0,1],[(1,3),(4,1)], normal_dist_2D_n);
df_normal_dist_2D = pd.DataFrame(normal_dist_2D, columns=["x", "y"]);
df_normal_dist_2D_sample = df_normal_dist_2D.sample(normal_dist_2D_n_sample);

# 2D distribution
dist3_n = 10000;
dist3_x = np.random.normal(size=dist3_n)
dist3_y = 2*dist3_x+3.5*np.random.normal(size=dist3_n)
#print(dist3_x)
#print("\n")
#print(dist3_y)
df_dist3_2D = pd.DataFrame({'x': dist3_x, 'y': dist3_y})
df_dist3_2D_sample = df_dist3_2D.sample(300);

# Dataset 4 columns
df4_n = 10000;
df4_x = np.random.normal(size=df4_n);
df4_y = 4*df4_x+3.5*np.random.normal(size=df4_n);
df4_z = df4_x+0.5*df4_y;
df4_w = df4_z*df4_x;
df4 = pd.DataFrame({'x': df4_x, 'y': df4_y, 'z': df4_z, 'w': df4_w})


'''
print(df_pokemon.head(10))
print("\n\n")
print(df_wine.head(10))
'''



##### UNIVARIATE PLOTTING #####

if (univariate_plotting):
    ### Using PANDAS library

    # BAR PLOT
    plt.figure()
    plt.title('plot.bar()')
    #df["type1"].value_counts().plot.bar()
    df_pokemon.type1.value_counts().plot.bar()
    
    # LINE PLOT
    plt.figure()
    plt.title('plot.line()')
    #print(df_pokemon.hp.value_counts())
    #print(df_pokemon.hp.value_counts().sort_index())
    df_pokemon.hp.value_counts().sort_index().plot.line()
    
    # HISTOGRAM
    plt.figure()
    plt.title('plot.hist()')
    df_pokemon.hp.plot.hist(bins=30)
    
    plt.figure()
    plt.title('plot.hist()')
    df_pokemon.weight_kg.plot.hist(bins=30)
    
    plt.figure()
    fig, ax = plt.subplots()
    plt.title('plot.hist() y log scale')
    df_pokemon.weight_kg.plot.hist(ax=ax, bins=30)
    ax.set_yscale('log')
    
    
    ### Using SEABORN library
    
    # COUNTPLOT  : bar chart
    plt.figure()
    plt.title('COUNTPLOT')
    #sns.countplot(df_wine.points)
    sns.countplot(df_wine['points'])
    
    # KDE    : kernel density estimate
    plt.figure()
    plt.title('KDE')
    #sns.kdeplot(df_wine.query('price < 200').price)
    sns.kdeplot(df_wine[df_wine['price'] < 200].price, shade=False)
    
    plt.figure()
    plt.title('KDE with different bandwidths')
    sns.kdeplot(df_wine[df_wine['price'] < 200].price, label='default')
    sns.kdeplot(df_wine[df_wine['price'] < 200].price, bw=.01, label="bw = 0.01")
    sns.kdeplot(df_wine[df_wine['price'] < 200].price, bw=.1, label="bw = 0.1")
    sns.kdeplot(df_wine[df_wine['price'] < 200].price, bw=.5, label="bw = 0.5")
    sns.kdeplot(df_wine[df_wine['price'] < 200].price, bw=1, label="bw = 1")
    plt.legend();
    
    # To be compared to a simple line chart:
    plt.figure()
    plt.title('compare to a simple plot.line()')
    df_wine[df_wine['price'] < 200]['price'].value_counts().sort_index().plot.line()
    
    # cut
    plt.figure()
    plt.title('plot line, KDE , rug plot')
    (df_pokemon.hp.value_counts().sort_index()/len(df_pokemon)).plot.line(label='line plot')
    sns.kdeplot(df_pokemon.hp, label='KDE', color='green')
    #sns.kdeplot(df_pokemon.hp, cut=0, label='cut')
    sns.rugplot(df_pokemon.hp.sort_index(), label='rug plot', color='red') 
    plt.legend();
                
    # DISTPLOT    : histogram and fit KDE
    plt.figure()
    plt.title('DISTPLOT: histogram and fit KDE')
    sns.distplot(normal_dist);
    
    plt.figure()
    plt.title('DISTPLOT: histogram, no KDE, rug plot')
    sns.distplot(normal_dist2, bins=15, hist=True, kde=False, rug=True);





##### BIVARIATE PLOTTING #####

if (bivariate_plotting):
    '''
    ### Using PANDAS library
    plt.figure()
    df_wine[df_wine['price'] < 200].sample(100).plot.scatter(x='points', y='price',color='b',edgecolor='w',s=80,marker='^')
    plt.title('PANDAS SCATTER PLOT')
    
    plt.figure()
    df_wine[df_wine['price'] < 200].sample(100).plot.scatter(x='points', y='price')
    plt.title('PANDAS SCATTER PLOT')
    
    plt.figure()
    df_wine[df_wine['price'] < 100].plot.hexbin(x='points', y='price', gridsize=18, sharex=False)
    plt.title('PANDAS HEXBIN PLOT')
    
    plt.figure()
    df_pokemon.sample(200).plot.scatter(x='defense', y='attack',color='C0',edgecolor='w',s=80,marker='o')
    plt.title('PANDAS SCATTER PLOT')
    
    plt.figure()
    df_pokemon.plot.hexbin(x='defense', y='attack', gridsize=15, sharex=False)
    plt.title('PANDAS HEXBIN PLOT')
    
    plt.figure()
    pokemon_stats_legendary = df_pokemon.groupby(['is_legendary', 'generation']).mean()[['attack', 'defense']]
    pokemon_stats_legendary.plot.bar(stacked=True)
    plt.title('PANDAS STACKED BAR PLOT')
    
    '''
    ### Using SEABORN library
    text_x1=-2
    text_y1=12
    text_x2=-2
    text_y2=17
    '''
    plt.figure()
    sns.kdeplot(df_wine[df_wine['price'] < 200].loc[:, ['price', 'points']].dropna().sample(5000))
    plt.title('SEABORN BIVARIATE KDE PLOT')
    
    # JointGrid class:
    
    plt.figure()
    lin_reg_fig = sns.JointGrid(x="x", y="y", data=df_dist3_2D_sample, ratio=3);
    lin_reg_fig = lin_reg_fig.plot_joint(plt.scatter, color="g") #,edgecolor='w',s=35,marker='^'
    lin_reg_fig = lin_reg_fig.plot_joint(sns.regplot, color="g")
    lin_reg_fig.ax_marg_x.hist(df_dist3_2D_sample["x"], color="b", alpha=.6, bins=15)
    lin_reg_fig.ax_marg_y.hist(df_dist3_2D_sample["y"], orientation="horizontal", color="b", alpha=.6, bins=15)
    plt.text(-3,text_y1,"JointGrid object\nplot_joint(plt.scatter) plot_joint(sns.regplot) \nax_marg_x.hist ax_marg_y.hist")
    
    plt.figure()
    lin_reg_fig = sns.JointGrid(x="x", y="y", data=df_dist3_2D, ratio=3);
    lin_reg_fig = lin_reg_fig.plot_joint(plt.hexbin, color="g", gridsize=30, edgecolor="white")
    #lin_reg_fig = lin_reg_fig.plot_joint(sns.regplot, color="g")
    lin_reg_fig = lin_reg_fig.plot_marginals(sns.kdeplot, shade=True, color='b')
    #plt.text(-3,text_y2,"JointGrid object \nplot_joint(plt.hexbin) \nplot_marginals(sns.kdeplot)")
    plt.colorbar()
    
    plt.figure()
    lin_reg_fig = sns.JointGrid(x="x", y="y", data=df_dist3_2D, ratio=3);
    lin_reg_fig = lin_reg_fig.plot_joint(sns.kdeplot, color="g", gridsize=18, edgecolor="white")
    lin_reg_fig = lin_reg_fig.plot_marginals(sns.kdeplot, shade=True, color='b')
    #plt.text(-3,text_y2,"JointGrid object \nplot_joint(sns.kdeplot) \nplot_marginals(sns.kdeplot)")
    
    plt.figure()
    lin_reg_fig = sns.JointGrid(x="x", y="y", data=df_dist3_2D, ratio=3);
    lin_reg_fig = lin_reg_fig.plot_joint(sns.kdeplot, color="g", shade=True, gridsize=18, edgecolor="white", cbar=True)
    lin_reg_fig = lin_reg_fig.plot_marginals(sns.kdeplot, shade=True, color='b')
    #plt.text(-3,text_y2,"JointGrid object \nplot_joint(sns.kdeplot) \nplot_marginals(sns.kdeplot)")
    
    '''
    # jointplot:
    plt.figure()
    #sns.axes_style({'patch.edgecolor': 'w','patch.force_edgecolor': True})
    sns.jointplot(x="x", y="y", data=df_dist3_2D_sample, ratio=3, kind='scatter',edgecolor='w',marker='o',s=80);
    plt.text(text_x2,text_y2,"jointplot(kind='scatter')")

    plt.figure()
    sns.jointplot(x="x", y="y", data=df_dist3_2D_sample, ratio=3, kind='reg');
    plt.text(text_x2,text_y2,"jointplot(kind='reg')")
    
    plt.figure()
    sns.jointplot(x="x", y="y", data=df_dist3_2D_sample, ratio=3, kind='resid');
    plt.text(text_x1,text_y1,"jointplot(kind='resid')")
    
    plt.figure()
    sns.jointplot(x="x", y="y", data=df_dist3_2D, ratio=3, kind='kde', cbar=True);
    
    plt.figure()
    sns.jointplot(x="x", y="y", data=df_dist3_2D, ratio=3, kind='hex');
    plt.colorbar()
    
    # PairGrid
    
    plt.figure()
    df4_pg = sns.PairGrid(df4)
    df4_pg.map_diag(sns.kdeplot)
    df4_pg.map_offdiag(sns.kdeplot, shade=True, n_levels=10, cbar=True);
    
    plt.figure()
    df4_sample = df4.sample(300)
    df4_pg = sns.PairGrid(df4_sample)
    df4_pg.map_diag(plt.hist)
    df4_pg.map_offdiag(plt.scatter,color='C0',edgecolor='w',s=70,marker='o');
    
    
    # FacetGrid
    
    facet_grid = sns.FacetGrid(df_pokemon, col="is_legendary", row="generation") #col_wrap=4
    facet_grid.map(plt.hist, "attack")


"""lin_reg_x = lin_reg.get_lines()[0].get_xdata()
lin_reg_y = lin_reg.get_lines()[0].get_ydata()
lin_reg_a = (lin_reg_y[len(lin_reg_y)-1]-lin_reg_y[0])/(lin_reg_x[len(lin_reg_x)-1]-lin_reg_x[0])
lin_reg_b = lin_reg_y[0] - lin_reg_a*lin_reg_x[0]
print(lin_reg_a)
print(lin_reg_b)
plt.text(-2, 10, str(lin_reg_a)+"x+"+str(lin_reg_b))
# y = ax +b
# b + Dy = y[0]   b = y[0] - a Dx = y[0] - ax[0]
#  Dy = a*Dx
"""


if(box_violin_plotting):

    ##### BOX PLOT #####
    
    plt.figure()
    boxplot_ax = sns.boxplot(x='variety', y='points', data=df_wine_top5)
    boxplot_ax.set_ylim([75, 105]) 
    
    ##### VIOLIN  PLOT #####
    
    plt.figure()
    #sns.violinplot(x='variety', y='points', data=df_wine[df_wine.variety.isin(df_wine.variety.value_counts()[:5].index)])
    sns.violinplot(x='variety', y='points', data=df_wine_top5)



##### MULTIVARIATE PLOTTING #####

if (multivariate_plotting):
    
    # Multivariate Scatter Plot
    
    plt.figure()
    sns.lmplot(x='points', y='price', hue='variety', markers=['o','D','d'],
               data=df_wine[(df_wine.variety.isin(df_wine.variety.value_counts().head(3).index)) & (df_wine.price < 200)].dropna().sample(600),
               fit_reg=True,
               scatter_kws={"s": 30}
              )
    
    # Multivariate Box Plot
    
    plt.figure()
    boxplot_ax = sns.boxplot(x='generation', y='attack', data=df_pokemon, hue='is_legendary')
    boxplot_ax.set_ylim([-50, 200]) 
    
    # Correlation Matrix
    
    plt.figure()
    corr_data = df4.corr()
    sns.heatmap(corr_data, annot=True)
    
    # Parallel coordinates
    
    #print(df_pokemon.iloc[0])
    plt.figure()
    df_pokemon_reduced = df_pokemon[df_pokemon.type1.isin(df_pokemon.type1.value_counts().head(3).index)]
    df_pokemon_reduced.rename(columns={'percentage_male': 'percentage\n_male'}, inplace=True)
    parallel_coordinates_data = df_pokemon_reduced[['height_m','weight_kg','percentage\n_male','attack','defense','speed','hp','type1']].dropna().sample(200)
    parallel_coordinates(parallel_coordinates_data, 'type1') #, palette=sns.color_palette("hls"))
    
    
    
if (time_series):
    plt.figure()
    df_shelter['date_of_birth'].value_counts().sort_values().plot.line()

    plt.figure()
    df_shelter['date_of_birth'].value_counts().resample('Y').sum().plot.line()

    #plt.figure()
    #df_shelter['date_of_birth'].value_counts().resample('Y').plot.line()
    
    
    
plt.show()
