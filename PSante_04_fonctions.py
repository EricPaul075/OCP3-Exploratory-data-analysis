#***********************************************************************************************************************
# Liste des fonctions de ce module :
#
# - listCategories(data, colName, *args, **kwargs)
#   --> regroupe les valeurs de la colonne (colName) en catégories, quantifie le nombre de valeurs pour chaque
#       catégorie, et affiche la liste des nRows lignes de tête (défaut=10) et nRows lignes de queue (si l'argument
#       'hnt' est spécifié) des catégories par ordre alphabétique (défaut) ou par taille décroissance (sort_by='size')
#       Retourne le dataframe de la liste des catégories si l'argument 'ret' est spécifié
#
# - changeValues(data, colName, value, otherValues)
#   --> remplace les valeurs de la liste (otherValues) de la colonne (colName) par la valeur (value)
#
# - mergeCol(data, col1, col2, **kwargs)
#   --> fusionne les colonnes (col1) et (col2) en fonction de l'argument (method='concat', 'best' par défaut)
#
# - reduce2selectedVar(data, selectedVar)
#   --> réduit le jeu de données à la liste des variables (selectedVar)
#
# - getProductFromBarCode(data)
#   --> donne des informations sur le produit correspondant au code barre s'il existe
#
# - univar_cat(data, feature)
#   --> représentation univariée de la variable catégorielle 'feature' de data sous forme de graphe secteurs
#       (faible nombre de catégories)
#
# - univar_big_cat(data, feature, nb_cat_display)
#   --> représentation univariée de la variable catégorielle'feature' de data sous forme de graphe secteurs et
#       histogramme pour les nb_cat catégories les plus représentées (variable à nombre élevé de catégories)
#
# - univar_combo_cat(data, feature1, feature2, n_graph=9)
#   --> représentation univariée de 2 variables catégorielles 'feature1' et 'features2', qui représente les
#       sous-catégories de 'feature1', sous forme de graphe à secteurs
#       un maximum de n_graph est représenté pour les sous-catégories
#
# - polyreg(data, xlabel, ylabel, full=False, deg=1)
#   --> Régression polynomiale de degré 'deg' à partir des données de 'data', avec x=colonne de nom 'xlabel'
#       et y=colonne de nom 'ylabel'
#       Renvoie les informations de la régression (par défaut les coefficient, sinon -full=True- également
#       les autres informations fournies par P.polyfit
#
# - pair_plot(data, pair, exclude_x=None, exclude_y=None, xmin=None, xmax=None, ymin=None, ymax=None)
#   --> représentation bivariée des 2 variables de 'data' spécifiée par 'pair'
#       si une donnée optionnelle est renseignée, représente un second graphique répondant à la spécification
#
# - eta_squared(x, y)
#   --> retourne le rapport de corrélation η² pour x=valeurs catégorielles et y=valeurs numériques
#
# - df_style_fct(val)
#   -> fonction de style (mapping) de display de dataframe: entoure en bords épais les cellules supérieures à un seuil
#
# - df_style(styler)
#   --> fonction de style (mapping) de display de dataframe: applique un gradient de couleur de fond selon les valeurs
#
# - cor_table(data, num_features, thresshold=0.5, df_style=None)
#   --> Etablit la table des coefficient de corrélation entre les variables numériques de 'data'
#       Applique df_style_fct avec 'thresshold' et df_style
#       Enregistre le cas échéant dans le fichier MS-Excel spécifié par df_style
#
# - eta_table(data, cat_features, num_features, thresshold=0.5, XLexport=None)
#   --> Etablit la table des rapports de corrélation entre les variables catégorielles (liste dans 'cat_features')
#       et numériques (liste dans 'num_features') de 'data'
#       Applique df_style_fct avec 'thresshold' et df_style pour l'affichage de table résultante
#       Enregistre le cas échéant dans le fichier MS-Excel spécifié par df_style
#
# - pair_boxplot(data, pair, nb_cat=5, cat_ordering='mean_num_values', exclude_x=None, xmin=None, xmax=None, save=None)
#   -> Représente le boxplot de la paire catégorielle / numérique avec options de classement des valeurs catégorielles
#      Si exclude_x, xmin ou xmax est spécifié, trace un second graphe avec cette spécification
#      Enregistre le cas échéant la figure dans le fichier spécifié par 'save'
#
# - welch_ttest(x, y)
#   --> renvoie le résultat du test de welch entre la variable catégorielle x et la variable numérique y
#       concernant sur l'égalité des moyennes de y pour chaque catégorie de x
#
# - anova(data, pair, nb_cat=5, alpha=0.05, save=None)
#   -> Etablit l'ANOVA de la paire catégorielle / numérique sur les nb_cat catégories les plus représentées
#      Teste les conditions de l'ANOVA avec le seuil 'alpha':
#       - test de normalité de la variable numérique sur chaque catégorie (Shapiro)
#       - test d'égalité des écarts types de la variable numérique sur chaque catégorie (Bartlett)
#       - si le test de Barlett est négatif, applique le test de Welch pour identifier les catégories pour
#         lesquelles les moyennes de la variable numérique ne sont pas significativement différentes
#       - test de Fischer (qui n'a de valeur qu'en fonction des tests ci-dessus)
#       - tracé du boxplot, des segments de droite reliant les moyennes et de la droite de régression des
#         valeurs moyennes - affiche l'équation de la droite de régression
#
#**************** Fonctions ACP et Classification sont issues du cours "Réalisez une          ***********************
#**************** analyse exploratoire de données", avec des modifications pour mon utilisation *********************
#
# - display_circle(ax, pcs, n_comp, pca, plan, c_labels=None, c_label_rotation=0, c_lims=None, filter=None)
#   --> Trace le cercle des corrélations dans les axes 'ax' spécifiés et pour le plan [Fi, Fj] spécifié (i,j de 1
#       à n_comp
#       Filtre les vecteurs de longueur inféreure à 'filter'
#
# - display_factorial_plan(ax, X_projected, n_comp, pca, plan, scale=None,
#                          p_labels=None, alpha=1, illustrative_var=None)
#   --> Trace la projection des individus dans les axes 'ax' spécifiés et pour le plan [Fi, Fj] spécifié (i,j de 1
#       à n_comp
#       Applique l'échelle 'scale' (ex: ['linear', 'syslog'] selon les échelles de Matplotlib
#
# - project_plot(ax, X_projected, n_comp, pca, plan, scale=None, p_labels = None, save=None)
#   --> Affiche la projection des individus sur le plan
#
# - circle_plot(ax, pcs, n_comp, pca, plan, c_labels=None, c_label_rotation=0, c_lims=None, filter=None, save=None)
#   --> Affiche le cercle des corrélations du plan spécifié
#
# - projetNcircle_plot(pcs, X_projected, n_comp, pca, plan, scale=None,
#                        p_labels = None, c_labels=None, c_label_rotation=0, c_lims=None, filter=None, save=None)
#   --> Affiche côté à côte le cercle des corrélations et la projection des individus du plan
#
# - display_scree_plot(pca)
#   --> Affichage Eboulis des valeurs propres avec le seuil du critère de Kaiser
#
# - plot_dendrogram(Z, names) - Non utilisée dans ce projet
#   --> Tracé du Dendrogramme
#
#******************************************** Autres fonctions *********************************************************
# - int_sup_div(x, y)
#   --> Renvoie la partie entière de la division x/y arrondie à l'entier supérieur
#
# - appli(data, association, keep=None, to_minimise=None, to_exclude=None, sort_by=None, to_display=None)
#   -> Illustration de l'application envisagée dans le cadre du projet
#***********************************************************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from numpy.polynomial import polynomial as P
import scipy.stats as st
from scipy.stats import f
import seaborn as sns
import re

# Display options
from IPython.display import (display, display_html, display_png, display_svg)
from colorama import Fore, Back, Style

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 199)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# Dossier contenant les figures
dossierFigures = ".\PSante_06_figures"


# Remove ILLEGAL CHARACTER to write DataFrame in a MS-Excel file
ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]|[\x00-\x1f\x7f-\x9f]|[\uffff]')
def illegal_char_remover(df):
    if isinstance(df, str):
        return ILLEGAL_CHARACTERS_RE.sub("", df)
    else:
        return df

# Ecriture dans un fichier MS-Excel → csv avec séparateur ';'
def writeDF2XLfile(data, fileName):
    #data4xl = data.applymap(illegal_char_remover)
    data.to_csv(fileName + ".csv", encoding='utf-8-sig', sep=';')


# Liste des catégories d'une variable et tri selon la variable
# sort_by peut prendre les valeurs suivantes:
#  - 'alpha' (défaut): tri aphanumérique
#  - 'size' : tri par la taille
# Arguments optionnels
#  - 'no_display': pas d'affichage de la liste des catégories
#  - 'hnt': affiche également les nRows dernières lignes de la liste des catégories
#  - 'ret': renvoie la liste des catégories et nombre d'individus associées
def listCategories(data, colName, *args, nRows=10, sort_by='alpha'):
    df = pd.DataFrame(data[colName])
    cat = df.groupby(sorted(df.columns.tolist()), as_index=False).size()
    if sort_by == 'size':
        cat.sort_values(by='size', ascending=False, inplace=True)
        cat.reset_index(inplace=True)
    if 'no_display' not in args:
        print("Liste des catégories de '", colName, "' ('size' indique le nombre d'occurrences) :", sep='')
        display(cat.head(int(nRows)))
        if 'hnt' in args:
            display(cat.tail(int(nRows)))
        print("-->", cat['size'].sum(), "éléments sur", df.shape[0], "soit",
              '{:0.2f}%'.format(100*cat['size'].sum()/df.shape[0]), '\n')
    if 'ret' in args:
        return cat

# Renomme avec la valeur 'value' toutes les valeurs contenues dans la liste otherList
def changeValues(data, colName, value, otherValues):
    mask = data[colName].isin(otherValues)
    data.loc[mask, colName] = value

#--------------------------------------------------------------------------------------------------
# Construction d'une variable combinant les informations disponibles pour catégoriser les produits
#
# Fonction pour fusionner 2 colonnes de data et renvoyer le nom de la colonne résultante
# avec 2 méthodes possibles :
# 1) method = 'concat' : concatène les valeurs des colonnes col1 et col2 sous forme de 2 chaines
#                        de caractères séparées par un ";"
# 2) method = 'best' (défaut) : détermine le meilleur des 2 colonnes selon les informations
#                               disponibles
# - col1: si col1 dispose de valeurs partout où col2 en dispose également
#         -> les valeurs de col1 sont retenues lorsque celles de col2 existent
# - col2: si col2 dispose de valeurs partout où col1 en dispose également
#         -> les valeurs de col2 sont retenues lorsque celles de col1 existent
# - concaténation dans une nouvelle colonne, à partir de col1, si les valeurs
#   sont en complémentarité
#   -> les valeurs de col1 sont retenues lorsque celles de col2 existent
def mergeCol(data, col1, col2, method='best'):
    if method == 'concat':
        print("* Fusion par concaténation des valeurs des colonnes", col1, "et", col2)
        concat = str(col1) + "_" + str(col2)
        data[concat] = data[col1].str.cat(data[col2], sep="; ", na_rep=None)
        return (concat)
    else:
        print("* Comparaison des colonnes", col1, "et", col2, "comprenant", data.shape[0], "lignes")
        df = pd.concat([data[col1], data[col2]], axis=1)
        nbCol1 = data.shape[0] - df[col1].isnull().sum(axis=0)
        nbCol2 = data.shape[0] - df[col2].isnull().sum(axis=0)
        print("  La colonne", col1, "contient", nbCol1, "valeurs et la colonne", col2, nbCol2, "valeurs")
        mask = pd.DataFrame([df.isnull().sum(axis=1)])
        nbConcat = mask.isin([0, 1]).sum(axis=1)[0]
        if nbConcat == max(nbCol1, nbCol2):
            if nbCol1 > nbCol2:
                print("  -> La fusion des colonnes", col1, "et", col2, "s'effectue préférentiellement avec", col1, '\n')
                return(col1)
            elif nbCol2 > nbCol1:
                print("  -> La fusion des colonnes", col1, "et", col2, "s'effectue préférentiellement avec", col2, '\n')
                return (col2)
            else:
                print("  -> La fusion des colonnes", col1, "et", col2,
                      "s'effectue indifféremment avec n'importe laquelle des 2 colonnes", '\n')
                return (col1)
        else:
            addVal = nbConcat - max(nbCol1, nbCol2)
            print("  -> La fusion des 2 colonnes à partir de", col1, "apporte", addVal, "valeurs supplémentaires", '\n')
            merge = str(col1) + "X" + str(col2)

            df.loc[~df[col1].isnull() & ~df[col2].isnull(), :] = np.nan
            df.dropna(how='all', inplace=True)
            df.sort_values(by=col2, inplace=True)
            display(df)

            data[merge] = data[col1].mask(data[col1].isnull(), data[col2])
            return(merge)


# Réduction du jeu de données aux colonnes présélectionnées
def reduce2selectedVar(data, selectedVar):
    print("\nRéduction du jeu de données aux variables :", selectedVar)
    dropColumns = []
    for col in data.columns.values.tolist():
        if col not in selectedVar:
            dropColumns.append(col)
    data.drop(dropColumns, axis=1, inplace=True)
    print("\nNouvelle dimension du jeu de données:", data.shape, '\n')


# Fonctionnement du code barre selon la norme GTIN (Global Trade Item Number),
# avec un code de longueur 8, 12, 13 ou 14
# La documentation précise que les codes sont EAN 13 ou des codes internes
# pour certains magasins (???). La structure du code EAN-13 :
# |o|o|o|f|f|f|f|p|p|p|p|p|c|
# |o|o|f|f|f|f|f|p|p|p|p|p|c|
# |1|2|3|4|5|6|7|8|9|A|B|C|D|
#  o : pays d'origine du produit, 300 à 379 pour la France
#  f : numéro du fabricant
#  p : numéro du produit du fabricant
#  c : clé de contrôle
# Recherche de code barre spécifique
def getProductFromBarCode(data):
    barCode = '20139315'
    while len(barCode) > 0:
        pdt = pd.DataFrame(data.loc[data['code'] == barCode])
        if not pdt.empty:
            if pdt['product_name'].values:
                print(" - Nom du produit :", pdt['product_name'].values[0])
            if pdt['product_category'].values:
                print(" - Catégorie :", pdt['product_category'].values[0])
            if pdt['nutrition_grade_fr'].values:
                print(" - Note nutritionnel :", str(pdt['nutrition_grade_fr'].values[0]).upper())
            if pdt['nutrition-score-fr_100g'].values:
                print(" - Score nutritionnel :", pdt['nutrition-score-fr_100g'].values[0])
        barCode = str(input("\nEntrez le code barre : "))



# Analyse univariée pour les variables à faible nombre de catégories
def univar_cat(data, feature):
    fig = plt.figure(figsize=(15,5))
    gs = GridSpec(nrows=1, ncols=2)
    fig.add_subplot(gs[0, 0])
    data[feature].value_counts(normalize=True, sort=True, ascending=True, dropna=False).plot(kind='pie')
    title = "Caractéristique " + feature
    plt.title(title)
    plt.ylabel(ylabel=None)
    fig.add_subplot(gs[0, 1])
    data[feature].value_counts(normalize=False, sort=True, ascending=True, dropna=False).plot(kind='bar')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Analyse univariée pour les variables à grand nombre de catégories
def univar_big_cat(data, feature, nb_cat_display):
    cat = listCategories(data, feature, 'ret', 'no_display', sort_by='size')
    nb_cat = cat.shape[0]
    print(Fore.BLUE + "", nb_cat, "catégories de", feature, "pour", data.shape[0], "produits")
    cat.drop('index', axis=1, inplace=True)
    print(" --> Affichage des", nb_cat_display-1,"premières catégories"+ Style.RESET_ALL)
    cat = cat.head(nb_cat_display)
    misc_size = data.shape[0] - cat['size'].sum()
    misc = pd.DataFrame([['autres', misc_size]], columns=cat.columns)
    cat = pd.concat([cat, misc], axis=0, ignore_index=True)
    cat.set_index(feature, inplace=True)
    # Tracé des graphiques
    fig = plt.figure(figsize=(15,5))
    gs = GridSpec(nrows=1, ncols=2)
    fig.add_subplot(gs[0, 0])
    cat['size'].plot(kind='pie')
    title = "Caractéristique " + feature
    plt.title(title)
    plt.ylabel(ylabel=None)
    fig.add_subplot(gs[0, 1])
    cat['size'][:nb_cat_display-1].plot(kind='bar')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Fonction de tracé combiné de 2 variables catégorielles pour lesquelles 'feature2'
# sont les sous-catégories de 'feature1'
# La fonction ordonne par nombre d'occurrences décroissantes les catégories 'feature1' puis 'feature2'
# La fonction trace le graphe par secteurs de 'feature1' puis trace les graphes par secteur de 'feature2'
# pour chaque catégorie de 'feature1' dans la limite de 'n_graph' sous-graphes (défaut=9)
def univar_combo_cat(data, feature1, feature2, n_graph=9):
    p1_cat = pd.DataFrame(data[feature1]).groupby(sorted(pd.DataFrame(data[feature1]).columns.tolist()),
                                                  as_index=False).size()
    p1_cat.sort_values(by='size', ascending=False, inplace=True)
    p1_cat.reset_index(inplace=True)
    list_p1_cat = p1_cat[feature1].values.tolist()
    # Liste combinée feature1 et feature2, triée par occurrence et filtrée des occurrences nulles
    df_cat = data[[feature1, feature2]].groupby(data[[feature1, feature2]].columns.tolist(), as_index=False).size()
    df_cat[feature1] = pd.Categorical(df_cat[feature1], list_p1_cat)
    df_cat.sort_values(by=[feature1, 'size'], ascending=[True, False], inplace=True)
    df_cat = df_cat[df_cat['size'] > 0]

    # Tracé des catégories feature1 et feature2
    # Organisation de l'ensemble des tracés
    nrows = min(int(1 + len(list_p1_cat)) + (1 + len(list_p1_cat)) % 2, n_graph)
    fig = plt.figure(figsize=(15, 5 * nrows))
    gs = GridSpec(nrows=nrows, ncols=2)
    # Tracé de feature1
    fig.add_subplot(gs[0, 0])
    data[feature1].value_counts(normalize=True, sort=True, ascending=True, dropna=False).plot(kind='pie')
    title = "Catactéristique " + feature1
    plt.title(title)
    plt.ylabel(ylabel=None)
    # Tracés de feature2 pour chaque catégorie de feature1
    n_cat = 1
    for cat in list_p1_cat:
        if n_cat <= n_graph:
            fig.add_subplot(gs[int(n_cat / 2), n_cat % 2])
            df = df_cat[df_cat[feature1] == cat]
            df = df.pivot_table(values='size', index=feature2, columns=feature1)
            df[cat].plot(kind='pie')
            title = feature2 + " pour " + cat
            plt.title(title)
            plt.ylabel(ylabel=None)
        n_cat += 1
    plt.tight_layout()
    plt.show()


# Régression avec np.polynomial.polynomial.polyfit entre les colonnes xlabel et ylabel du DataFrame data
# Si l'argument 'full' n'est pas spécifié, renvoie les coefficients du polynôme: poly[i] coefficient du terme en X^i
# Sinon retourne 'reg' avec:
# - poly=reg[0] coefficients du polynôme: poly[i] coefficient du terme en X^i
# - list=reg[1] permet de calculer r2 = 1 - list[0][0] / (np.var(yData) * len(yData))
def polyreg(data, xlabel, ylabel, full=False, deg=1):
    xData = data[xlabel].copy(deep=True)
    yData = data[ylabel].copy(deep=True)
    if not full: return P.polyfit(xData, yData, deg=deg)
    else: return P.polyfit(xData, yData, deg=deg, full=True)



# Tracé de graphe par paire de variables numériques avec :
# - à gauche le nuage de l'ensemble des points
# - à droite, si un ou plusieurs arguments sont spécifiés, le nuage réduit avec la courbe de tendance
def pair_plot(data, pair, exclude_x=None, exclude_y=None, xmin=None, xmax=None, ymin=None, ymax=None):
    # Initialisation des paramètres avec les kwargs
    if exclude_x is not None: bx=True
    else: bx=False
    if exclude_y is not None: by=True
    else: by=False
    if xmin is not None: bxmin=True
    else: bxmin=False
    if xmax is not None: bxmax=True
    else: bxmax=False
    if ymin is not None: bymin=True
    else: bymin=False
    if ymax is not None: bymax=True
    else: bymax=False

    df = data[pair].copy(deep=True)
    coef_p = st.pearsonr(df[pair[0]], df[pair[1]])[0]
    # Pour éventuellement examiner la corrélation en excluant les valeurs nulles
    if not bx and not by and not bxmin and not bxmax and not bymin and not bymax:
        print("Nuage complet, Pearson=" + f"{coef_p:.2f}")
        sns.jointplot(data=df, x=pair[0], y=pair[1], kind="reg", marginal_kws=dict(bins=20, fill=True))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        suptitle = "Pairplot " + pair[0] + " et " + pair[1]
        fig.suptitle(suptitle)
        sns.scatterplot(ax=axes[0], data=df, x=pair[0], y=pair[1])
        title = "Nuage complet, Pearson=" + f"{coef_p:.2f}"
        axes[0].set_title(title)
        if bx:
            df = df[df[pair[0]] != exclude_x]
        if by:
            df = df[df[pair[1]] != exclude_y]
        if bxmin:
            df = df[df[pair[0]] >= xmin]
        if bxmax:
            df = df[df[pair[0]] <= xmax]
        if bymin:
            df = df[df[pair[1]] >= ymin]
        if bymax:
            df = df[df[pair[1]] <= ymax]
        coef_p_ve = st.pearsonr(df[pair[0]], df[pair[1]])[0]
        g = sns.scatterplot(ax=axes[1], data=df, x=pair[0], y=pair[1])
        poly = polyreg(df, pair[0], pair[1])
        g.axline(xy1=(0, poly[0]), slope=poly[1], color='b', dashes=(5, 2))
        title = "Nuage partiel, Pearson=" + f"{coef_p_ve:.2f}"
        axes[1].set_title(title)
    plt.tight_layout()
    plt.show()
    # plt.savefig("Figure - pairplot recherche correlation.png", dpi=150)


# Calcul du rapport de corrélation entre une variable catégorielle x et une variable quantitative y
def eta_squared(x, y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


# Fonctions de style d'affichage de dataframe avec display
# Code type: df.style.pipe(df_style).applymap(df_style_fct)
# Seuil de mise en évidence de valeurs, à ajuster dans le Notebook
thresshold = 0.5
# Ajoute une bordure rouge épaisse aux valeurs supérieures au seuil
def df_style_fct(val):
    border = '3px solid red' if val > thresshold else '1px solid black'
#    color = 'green' if val >= thresshold else None
#    return 'color: %s' % color
#    return 'background-color: %s' % color
    return 'border: %s' % border
# Règle le style d'affichage du dataframe
# - Nombres float avec 2 chiffres après la virgule
# - Gradient de couleurs avec la colormap "BuGn" entre les valeurs vmin et vmax
def df_style(styler):
    styler.format("{:.2f}")
    styler.background_gradient(axis=None, vmin=0, vmax=1, cmap="BuGn")
    return styler


def cor_table(data, num_features, thresshold=0.5, XLexport=None):
    # Etablissement de la table des corrélation entre variables numériques
    df_cor = np.array([st.pearsonr(data[feature], data[other_feature])[0] for feature in num_features for other_feature in num_features])
    df_cor = pd.DataFrame(np.reshape(df_cor, (len(num_features), len(num_features))), index=num_features, columns=num_features)

    # Export optionnel de la table dans un fichier excel
    if XLexport is not None:
        writeDF2XLfile(df_cor, XLexport)

    # Filtrage avec la valeur de seuil
    df_hit = df_cor.applymap(lambda x: x >= thresshold)
    print(Fore.LIGHTGREEN_EX + "* Recherche de corrélations potentielle entre les caractéristiques numériques",
          ", coefficient de Pearson >", thresshold, ":" + Style.RESET_ALL)
    list_cor = []
    list_pair_cor = []
    for feature in num_features:
        for other_feature in df_hit.index[df_hit[feature]==True].tolist():
            if other_feature != feature:
                if other_feature not in list_cor:
                    list_cor.append(other_feature)
                if [other_feature, feature] not in list_pair_cor:
                    list_pair_cor.append([feature, other_feature])
        df_cor.at[feature, feature] = 0
    df_cor = df_cor.loc[list_cor, list_cor]
    # Affichage du résultat
    print(Fore.LIGHTBLUE_EX + "  -> Liste des", len(list_pair_cor), "paires de caractéristiques à examiner :" + Style.RESET_ALL)
    display(list_pair_cor)
    # Affichage des correlations en surbrillance
    print(Fore.LIGHTBLUE_EX + "  -> Table de corrélation :" + Style.RESET_ALL)
    display(df_cor.style.pipe(df_style).applymap(df_style_fct))


# Etablissement d'une table de calcul des η² par paires de variables catégorielles,
# dont la liste est dans cat_features, et numériques, dont la liste est dans num_features
# La valeur de seuil (thresshold) filtre les valeurs supérieures
# Possibilité d'export de la table vers MS-Excel en spécifiant le nom de fichier
def eta_table(data, cat_features, num_features, thresshold=0.5, XLexport=None):
    # Matrice des corrélations
    df_eta = np.array(
        [eta_squared(data[feature], data[other_feature]) for feature in cat_features for other_feature in
         num_features])
    df_eta = pd.DataFrame(np.reshape(df_eta, (len(cat_features), len(num_features))), index=cat_features,
                          columns=num_features)

    # Export vers MS-Excel avec le fichier spécifié dans 'save'
    if XLexport is not None:
        writeDF2XLfile(df_eta, XLexport)

    # Test de la matrice par rapport à un seuil
    df_hit = df_eta.applymap(lambda x: x >= thresshold)
    print(
        Fore.LIGHTGREEN_EX + "* Recherche de corrélations potentielle entre caractéristiques catégorielles et numériques",
        ", coefficient η² >", thresshold, ":" + Style.RESET_ALL)

    # Liste des paires avec η² > thresshold et liste des exclusions par overfitting (trop de catégories)
    list_pair_eta = []
    cat_exclusions = []
    thr_cat_excl = int(data.shape[0] / 100)
    for num_feature in num_features:
        for cat_feature in df_hit.index[df_hit[num_feature] == True].tolist():
            if len(data[cat_feature].unique()) > thr_cat_excl:
                if cat_feature not in cat_exclusions:
                    cat_exclusions.append(cat_feature)
            elif [cat_feature, num_feature] not in list_pair_eta:
                list_pair_eta.append([cat_feature, num_feature])
    list_pair_eta.sort()

    # Affichage des exclusions par overfitting
    if len(cat_exclusions) > 0:
        print(Fore.LIGHTBLUE_EX + "  -> Caractéristiques exclues de l'analyse car comportant trop (>",
              thr_cat_excl, ", soit >{:.1f}%) de catégories (situation d'overfitting):".format(
                round(100.0 * thr_cat_excl / data.shape[0], 1)) +
              Style.RESET_ALL)
        display(cat_exclusions)
        display(df_eta.loc[cat_exclusions].style.pipe(df_style).applymap(df_style_fct))

    # Affichage des paires de caractéristiques au-dessus du seuil de corrélation
    print(Fore.LIGHTBLUE_EX + "  -> Liste des", len(list_pair_eta), "paires de caractéristiques à examiner :" +
          Style.RESET_ALL)
    display(list_pair_eta)
    cat_items = list(set([item[0] for item in list_pair_eta]))
    num_items = list(set([item[1] for item in list_pair_eta]))
    df_eta = df_eta.loc[cat_items, num_items]
    print(Fore.LIGHTBLUE_EX + "\n  -> Table de corrélation :" + Style.RESET_ALL)
    display(df_eta.style.pipe(df_style).applymap(df_style_fct))



# Tracé de graphe boxplot horizontal, simple ou par paire avec :
# - à gauche le nuage de l'ensemble des points
# - à droite, si un ou plusieurs arguments sont spécifiés, le nuage réduit avec la courbe de tendance
# Paramètres:
# - data: dataframe contenant les données en colonne
# - pair: paire de labels de colonne [col_label_cat, col_label_num]
# Arguments:
# - nb_cat= nombre de catégories à afficher, 5 par défaut
# - cat_ordering=ordre de présentation des catégories:
#       . mean_num_values (défaut): par moyenne décroissante des valeurs numériques associées
#       . cat_frequency: par fréquence décroissante des valeurs de catégorie
#       . alpha: par ordre alphabétique
# - exclude_x: exclut cette valeur numérique du nuage
# - xmin: valeur numérique minimum affichée
# - xmax: valeur numérique maximum affichée
def pair_boxplot(data, pair, nb_cat=5, cat_ordering='mean_num_values',
                 exclude_x=None, xmin=None, xmax=None, save=None):
    # Initialisation des paramètres
    if exclude_x is not None: bx=True
    else: bx=False
    if xmin is not None: bxmin=True
    else: bxmin=False
    if xmax is not None: bxmax=True
    else: bxmax=False

    df = data[pair].copy(deep=True)

    if cat_ordering == 'alpha':
        list_cat = sorted(df[pair[0]].unique())
    elif cat_ordering == 'cat_frequency':
        # Données filtrées sur les nb_cat plus fréquentes catégories
        list_cat = df[pair[0]].value_counts().head(nb_cat).index.values.tolist()
    else:
        # Données filtrées sur les nb_cat pour lesquelles la moyenne des valeurs numériques est la plus élevée
        df_cat = df.groupby(pair[0], as_index=False).agg(means=(pair[1], "mean")).sort_values(by='means', ascending=False)
        list_cat = df_cat[pair[0]].head(nb_cat).values.tolist()

    df.drop(index=df.loc[~df[pair[0]].isin(list_cat), :].index, inplace=True)
    df[pair[0]] = pd.Categorical(df[pair[0]], categories=list_cat, ordered=True)

    eta_sqr = eta_squared(df[pair[0]], df[pair[1]])
    fig_h = nb_cat if nb_cat<6 else int((5*nb_cat+40)/15)

    # Propriétés graphiques
    medianprops = {'color': "black"}
    meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}

    if not bx and not bxmin and not bxmax:
        print("Nuage complet pour les", nb_cat, "catégories du graphique, η²=" + f"{eta_sqr:.2f}")
        rcParams['figure.figsize'] = 10, fig_h
        sns.boxplot(x=pair[1], y=pair[0], data=df, showfliers=True,
                    medianprops=medianprops, showmeans=True, meanprops=meanprops)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, fig_h), sharey=True)
        suptitle = "Boxplot " + pair[0] + " et " + pair[1]
        fig.suptitle(suptitle)
        sns.boxplot(ax=axes[0], x=pair[1], y=pair[0], data=df, showfliers=True,
                    medianprops=medianprops, showmeans=True, meanprops=meanprops)
        title = "Nuage complet pour les" + str(nb_cat) + "catégories du graphique, η²=" + f"{eta_sqr:.2f}"
        axes[0].set_title(title)
        if bx:
            df = df[df[pair[1]] != exclude_x]
        if bxmin:
            df = df[df[pair[1]] >= xmin]
        if bxmax:
            df = df[df[pair[1]] <= xmax]
        eta_sqr_ve = eta_squared(df[pair[0]], df[pair[1]])
        sns.boxplot(ax=axes[1], x=pair[1], y=pair[0], data=df, showfliers=True,
                    medianprops=medianprops, showmeans=True, meanprops=meanprops)
        title = "Nuage partiel, η²=" + f"{eta_sqr_ve:.2f}"
        axes[1].set_title(title)
    plt.tight_layout()
    plt.show()
    if save is not None:
        plt.tight_layout()
        plt.show()
        plt.savefig(save, dpi=150)

# Test de Welch, retours:
# - dof= degrés de liberté
# - t  = test de Welch
# - p  = p-value
def welch_ttest(x, y):
    dof = (x.var() / x.size + y.var() / y.size) ** 2 / (
            (x.var() / x.size) ** 2 / (x.size - 1) + (y.var() / y.size) ** 2 / (y.size - 1))
    t, p = st.ttest_ind(x, y, equal_var=False)
    return t, p, dof

def anova(data, pair, nb_cat=5, alpha=0.05, save=None):
    df = data[pair].copy(deep=True)

    # Filtrage des catégories qui contiennent moins de 'Nb_prod_per_cat_min' lignes (min=3)
    df_cat = df.groupby(pair[0], as_index=False).agg(means=(pair[1], "mean"), size=(pair[0], "size")).sort_values(
        by='means', ascending=False).reset_index(drop=True)
    Nb_prod_per_cat_min = 3
    list_cat = df_cat.loc[df_cat['size'] >= Nb_prod_per_cat_min, pair[0]].tolist()
    df.drop(index=df.loc[~df[pair[0]].isin(list_cat), :].index, inplace=True)
    df_cat.drop(index=df_cat.loc[~df_cat[pair[0]].isin(list_cat), :].index, inplace=True)
    df_cat.reset_index(drop=True, inplace=True)

    # Filtrage sur les nb_cat pour lesquelles la moyenne des valeurs numériques est la plus élevée
    df_cat = df_cat.head(nb_cat)
    list_cat = df_cat[pair[0]].head(nb_cat).values.tolist()
    nb_cat = min(nb_cat, len(list_cat))
    df.drop(index=df.loc[~df[pair[0]].isin(list_cat), :].index, inplace=True)
    df[pair[0]] = pd.Categorical(df[pair[0]], categories=list_cat, ordered=True)

    # Calcul du rapport de corrélation
    eta_sqr = eta_squared(df[pair[0]], df[pair[1]])
    print("Rapport de corrélation pour les k=", nb_cat, "catégories du graphique et n=",
          df.shape[0], "données : η²=" + f"{eta_sqr:.2f}")

    # Remplacement des catégories par une valeur numérique pour le test
    df['cat'] = df[pair[0]].copy()
    df['cat'] = df['cat'].astype("category")
    df['cat'].replace(df['cat'].cat.categories, [i for i in range(0, len(df['cat'].cat.categories))], inplace=True)
    df['cat'] = df['cat'].astype("int")

    # Tests sur les variables
    # Test sur la condition de loi normale
    tn = True
    for cat in range(nb_cat):
        ts = st.shapiro(df.loc[df['cat']==cat, pair[1]].values)
        tn = tn and (ts.pvalue<=alpha)
        if ts.pvalue>alpha:
            print(" -> Test de normalité de Shapiro négatif sur la catégorie", cat, " :",
                  f"W= {ts.statistic:.2f}, p-value= {ts.pvalue:.2e}")
    if tn:
        print(" -> Test de normalité de Shapiro positif pour toutes les catégories")

    # Test d'homoscédasticité (écarts types égaux entre les catégories)
    gb = df.groupby(pair[0])[pair[1]]
    stat, p_bartlett = st.bartlett(*[gb.get_group(x).values for x in gb.groups])
    if p_bartlett<=alpha:
        print(f" -> Test d'homoscédasticité de Bartlett négatif : p-value={p_bartlett:.2e}")
        np.set_printoptions(precision=3)
        print("    Ecarts types:", np.array([gb.get_group(x).values.std() for x in gb.groups]))
    else:
        print(" -> Test d'homoscédasticité de Bartlett positif (écarts types égaux entre les catégories):",
              f"p_bartlett={p_bartlett:.2e}")

    # Test de Welch, dans le cas où le test d'homoscédasticité est négatif
    tw = True
    list = []
    if p_bartlett <= alpha:
        for x in gb.groups:
            for y in gb.groups:
                if x!=y and ([x, y] not in list):
                    list.append([x, y])
                    list.append([y, x])
                    t, p, dof = welch_ttest(gb.get_group(x).values, gb.get_group(y).values)
                    tw = tw and (p<=alpha)
                    if p>alpha:
                        print(" -> Test de Welch négatif entre les catégories '", x, "' et '", y, "' :",
                              f"W={t:.2f}, p-value={p:.3f}, dof={dof:.2f}")
        if tw:
            print(" -> Test de Welch (non égalité des moyennes) positif pour toutes les catégories")

    # Test statistique
    dfn = nb_cat - 1
    dfd = df.shape[0] - nb_cat
    F_crit = f.ppf(1 - alpha, dfn, dfd)
    F_stat, p = st.f_oneway(df['cat'], df[pair[1]])
    sign_F = ">" if F_stat > F_crit else "<"
    sign_p = ">" if p > alpha else "<"
    if (sign_F==">") and (sign_p=="<"):
        res_test = "positif"
    else:
        res_test = "négatif"
    print(f"Résultat {res_test} du test de Fisher : F={F_stat:.2f} {sign_F} {F_crit:.2f}",
          f" , et p-value={p:.2e} {sign_p} {alpha:0.2f}")
    print(Fore.LIGHTBLACK_EX + " Rappel des hypothèses relatives au test :")
    print("  - H0 : les moyennes par catégories sont égales entre elles (les variables sont indépendantes)")
    print("  - H1 : la moyenne d'au moins une catégorie diffère des autres (les variables sont corrélées)" +
          Style.RESET_ALL)

    fig_h = nb_cat if nb_cat<6 else int((5*nb_cat+40)/15)

    # Propriétés graphiques
    medianprops = {'color': "black"}
    meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}

    rcParams['figure.figsize'] = 10, fig_h
    sns.boxplot(x=pair[1], y=pair[0], data=df, showfliers=False,
                medianprops=medianprops, showmeans=True, meanprops=meanprops)

    # Tracé des lignes reliant les valeurs moyennes de chaque catégorie
    plt.plot(df_cat.means.values, df_cat.index.values, linestyle='--', c='#000000')

    # Régression linéaire sur les valeurs moyennes
    reg = P.polyfit(df_cat.means.values, df_cat.index.values, deg=1, full=True)
    yPredict = P.polyval(df_cat.means.values, reg[0])
    coef_cor = 1 - reg[1][0][0] / (np.var(df_cat.index.values) * len(df_cat.index.values))
    a = -1 / reg[0][1]
    mu = -reg[0][0] / reg[0][1] - a * (df_cat.shape[0] - 1)
    sign = '+' if a >= 0 else '-'
    print(f"\nMoyenne catégorielle : '{pair[1]}' = {mu:.2f}  {sign} {abs(a):.2f} * '{pair[0]}', avec :",
          f"'{df_cat[pair[0]][df_cat.shape[0] - 1]}'= 0 , …, '{df_cat[pair[0]][0]}'= {df_cat.shape[0] - 1}")
    print(f" -->Coefficient de corrélation r² ={coef_cor:.2f}")

    # Tracé de la droite de régression linéaire
    plt.plot(df_cat.means.values, yPredict, linewidth=2, linestyle='-', c='#FF0000')

    plt.tight_layout()
    plt.show()
    if save is not None:
        plt.savefig(save, dpi=150)


# *********************************************************************************************************************
# Fonctions pour ACP
#
# * display_circle(ax, pcs, n_comp, pca, plan, c_labels=None, c_label_rotation=0, c_lims=None, filter=None)
#   affiche le cercle des corrélations du plan=[i,j] avec i et j dans [1, n_comp] et i≠j
#   en fixant optionnellement c_lims = xmin, xmax, ymin, ymax
#   en filtrant optionnellement les vecteurs de longueur inférieurs à filter
#
#
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram


def display_circle(ax, pcs, n_comp, pca, plan, c_labels=None, c_label_rotation=0, c_lims=None, filter=None):
    d1 = plan[0] - 1
    d2 = plan[1] - 1
    if (d1 not in range(0, n_comp)) or (d1 not in range(0, n_comp)):
        print("Numéro d'axe d'inertie or limite")
        return

    if filter is not None:
        b = (pcs[d1, :] * pcs[d1, :] + pcs[d2, :] * pcs[d2, :]) > filter**2
        pcs[d1, :] = pcs[d1, :] * b
        pcs[d2, :] = pcs[d2, :] * b

    # détermination des limites du graphique
    if c_lims is not None:
        xmin, xmax, ymin, ymax = c_lims
    elif pcs.shape[1] > 30:
        xmin, xmax, ymin, ymax = -1, 1, -1, 1
    else:
        e = 0.0
        xmin, xmax, ymin, ymax = min(pcs[d1, :])-e, max(pcs[d1, :])+e, min(pcs[d2, :])-e, max(pcs[d2, :])+e
        xmin, xmax, ymin, ymax = max(min(xmin,0),-1), min(max(0,xmax),1), max(min(ymin,0),-1), min(max(0,ymax),1)


        # affichage des flèches
        # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
    if pcs.shape[1] < 30:
        ax.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1, :], pcs[d2, :],
                   angles='xy', scale_units='xy', scale=1, color="grey")
        # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
    else:
        lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
        ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

        # affichage des noms des variables

    if c_labels is not None:
        for i, (x, y) in enumerate(pcs[[d1, d2]].T):
            #if (x != 0 and y != 0) and x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            if not(x==0 and y==0):
                ax.text(x, y, c_labels[i],
                        fontsize='10', ha='center', va='center', rotation=c_label_rotation, alpha=1)

    # affichage du cercle
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
    ax.set_aspect(1)
    ax.add_artist(circle)

    # définition des limites du graphique
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # affichage des lignes horizontales et verticales
    ax.plot([-1, 1], [0, 0], color='grey', ls='--')
    ax.plot([0, 0], [-1, 1], color='grey', ls='--')

    # nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel('F{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
    ax.set_ylabel('F{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

    ax.set_title("Cercle des corrélations (F{} et F{})".format(d1 + 1, d2 + 1))



def display_factorial_plan(ax, X_projected, n_comp, pca, plan, scale=None,
                           p_labels=None, alpha=1, illustrative_var=None):
    d1 = plan[0]-1
    d2 = plan[1]-1
    if (d1 not in range(0, n_comp)) or (d1 not in range(0, n_comp)):
        print("Numéro d'axe d'inertie hors limite")
        return

    # affichage des points
    if illustrative_var is None:
        ax.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
    else:
        illustrative_var = np.array(illustrative_var)
        for value in np.unique(illustrative_var):
            selected = np.where(illustrative_var == value)
            ax.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
        ax.legend()

    # affichage des labels des points
    if p_labels is not None:
        for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
            ax.text(x, y, p_labels[i], fontsize='10', ha='center', va='center')

    # détermination des limites du graphique
    boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
    ax.set_xlim(-boundary, boundary)
    ax.set_ylim(-boundary, boundary)

    # détermination de l'échelle du graphique
    if scale is None:
        scalex = 'linear'
        scaley = 'linear'
    else:
        scalex, scaley = scale
    ax.set_xscale(scalex)
    ax.set_yscale(scaley)

    # affichage des lignes horizontales et verticales
    ax.plot([-100, 100], [0, 0], color='grey', ls='--')
    ax.plot([0, 0], [-100, 100], color='grey', ls='--')

    # nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel("F{i} ({var}%) - ech: '{sc}'".format(i=d1+1,
                                                    var=round(100 * pca.explained_variance_ratio_[d1], 1),
                                                    sc=str(scalex)))
    ax.set_ylabel("F{i} ({var}%) - ech: '{sc}'".format(i=d2 + 1,
                                                    var=round(100 * pca.explained_variance_ratio_[d2], 1),
                                                    sc=str(scaley)))

    ax.set_title("Projection des individus (sur F{} et F{})".format(d1 + 1, d2 + 1))

    #ax.show(block=False)


def project_plot(ax, X_projected, n_comp, pca, plan, scale=None, p_labels = None, save=None):
    out = display_factorial_plan(ax, X_projected, n_comp, pca, plan, scale=scale, p_labels = p_labels)
    if save is not None:
        plt.savefig(save, dpi=150)
    return out

def circle_plot(ax, pcs, n_comp, pca, plan, c_labels=None, c_label_rotation=0, c_lims=None, filter=None, save=None):
    out = display_circle(ax, pcs, n_comp, pca, plan,
                         c_labels=c_labels, c_label_rotation=c_label_rotation, c_lims=c_lims, filter=filter)
    if save is not None:
        plt.savefig(save, dpi=150)
    return out

def projetNcircle_plot(pcs, X_projected, n_comp, pca, plan, scale=None,
                       p_labels = None, c_labels=None, c_label_rotation=0, c_lims=None, filter=None, save=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Cercle des corrélation et projection dans le plan factoriel F{}-F{}".format(plan[0],plan[1]),
                 fontweight='bold')
    project_plot(axs[0], X_projected, n_comp, pca, plan, scale=scale, p_labels = p_labels)
    circle_plot(axs[1], pcs, n_comp, pca, plan,
                c_labels=c_labels, c_label_rotation=c_label_rotation, c_lims=c_lims, filter=filter)
    if save is not None:
        print("  --> Sauvegarde du tracé dans:", save)
        plt.savefig(save, dpi=150)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_ * 100
    plt.bar(np.arange(len(scree)) + 1, scree, label='Inertie par composante')
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker='o', label='Cumul inertie')
    kaiser_crit = 100 / np.shape(pca.components_)[1]
    plt.plot([1, len(scree)], [kaiser_crit, kaiser_crit], c='green', label='Critère de Kaiser')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.legend(loc='center right')
    plt.title("Eboulis des valeurs propres")
    plt.tight_layout()
    plt.show(block=False)


def plot_dendrogram(Z, names):
    plt.figure(figsize=(10, 25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels=names,
        orientation="left",
    )
    plt.show()

def int_sup_div(x, y):
    if x%y==0: return int(x/y)
    else: return int(1+x/y)


def appli(data, association, keep=None, to_minimise=None, to_exclude=None, sort_by=None, to_display=None):
    df = data.copy()

    # Filtrage des données en conservant 'keep' et au moins une des autres variables de 'association'
    if keep is not None:
        df['not_keep'] = df[keep]
        for item in association:
            if item != keep:
                df['not_keep'] = df['not_keep'] + df[item]
            else:
                df['not_keep'] = df['not_keep'] - df[item]
        df = df[(df['not_keep'] != 0) & (df[keep] != 0)]
        df.drop(columns='not_keep', inplace=True)
    else:
        for item in association:
            df = df[df[item] != 0]

    # Exclusion du composant 'to_exclude'
    if to_exclude is not None:
        for i in range(len(to_exclude[0])):
            df.drop(index=df.loc[df[to_exclude[0][i]] == str(to_exclude[1][i]), :].index, inplace=True)

    # Tri des données selon 'nutrition-score-fr_100g' et 'to_minimise' croissant
    subset = sort_by[0] if sort_by is not None else []
    ascending = sort_by[1] if sort_by is not None else []

    if to_minimise is not None:
        for item in to_minimise:
            subset.append(item)
            ascending.append(True)

    if (sort_by is not None) or (to_minimise is not None):
        df.sort_values(by=subset, ascending=ascending, inplace=True)

    # Préparation du jeu de données de sortie
    if to_display is not None:
        columns = to_display.copy()
        columns.extend(association)
    else:
        columns = df.columns.tolist()
    return df[columns]