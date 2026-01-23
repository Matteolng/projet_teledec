import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal, ogr

# Bibliothèques Scikit-Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D

# Bibliothèque interne SIGMA
from libsigma import read_and_write as rw
from libsigma import classification as cla

# =============================================================================
# ETAPE 1 : PREPARATION DES DONNEES ET ANALYSE
# =============================================================================

def rasterize_shapefile(image_ref_path, shp_path, output_raster, attribute_col='strate'):
    """
    Transforme un shapefile en raster aligné sur une image de référence.
    """
    raster_ds = gdal.Open(image_ref_path)
    geotransform = raster_ds.GetGeoTransform()
    projection = raster_ds.GetProjection()
    x_size = raster_ds.RasterXSize
    y_size = raster_ds.RasterYSize
    raster_ds = None 

    driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(output_raster):
        os.remove(output_raster)
        
    target_ds = driver.Create(output_raster, x_size, y_size, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)
    
    shp_ds = ogr.Open(shp_path)
    layer = shp_ds.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], layer, options=[f"ATTRIBUTE={attribute_col}"])

    target_ds.FlushCache()
    target_ds = None
    shp_ds = None
    print(f"Rasterisation terminée : {output_raster}")

def plot_poly_counts(shp_path, col_classe, output_path):
    """
    Affiche et sauvegarde le nombre de polygones par strate.
    """
    gdf = gpd.read_file(shp_path)
    counts_poly = gdf[col_classe].value_counts().sort_index()
    colors_dict = {1: 'brown', 2: 'green', 3: 'purple', 4: 'darkgreen'}
    colors_list = [colors_dict.get(i, 'gray') for i in counts_poly.index]

    plt.figure(figsize=(8, 5))
    plt.bar(counts_poly.index.astype(str), counts_poly.values, color=colors_list)
    plt.title("Nombre de polygones par strate")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.show()

def plot_pixel_counts(raster_path, output_path):
    """
    Affiche et sauvegarde la répartition des pixels par strate.
    """
    arr = rw.load_img_as_array(raster_path)
    classes, pixel_counts = np.unique(arr, return_counts=True)
    mask = classes != 0
    classes_reelles = classes[mask]
    counts_reels = pixel_counts[mask]
    colors_dict = {1: 'brown', 2: 'green', 3: 'purple', 4: 'darkgreen'}

    plt.figure(figsize=(8, 5))
    colors_list_pix = [colors_dict.get(i, 'gray') for i in classes_reelles]
    plt.bar(classes_reelles.astype(str), counts_reels, color=colors_list_pix)
    plt.title("Nombre de pixels par strate")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.show()

# =============================================================================
# ETAPE 2 : CALCULS D'INDICES ET PHENOLOGIE (NARI)
# =============================================================================

def process_nari_phenology(base_path, dates, output_fig_path):
    """
    Analyse temporelle de l'indice NARI par strate.
    """
    path_b03 = os.path.join(base_path, 'pyrenees_24-25_B03.tif')
    path_b05 = os.path.join(base_path, 'pyrenees_24-25_B05.tif')
    output_raster = os.path.join(base_path, 'sample_strata.tif')

    nari_means = {1: [], 2: [], 3: [], 4: []}
    nari_stds = {1: [], 2: [], 3: [], 4: []}

    arr_samples = rw.load_img_as_array(output_raster)
    arr_samples_2d = arr_samples[:, :, 0] if arr_samples.ndim == 3 else arr_samples

    ds_b03 = rw.open_image(path_b03)
    ds_b05 = rw.open_image(path_b05)
    if ds_b03 is None or ds_b05 is None:
        raise FileNotFoundError("Fichiers B03 ou B05 introuvables.")

    nb_bandes = ds_b03.RasterCount
    
    for i in range(min(len(dates), nb_bandes)):
        idx_gdal = i + 1
        b03 = ds_b03.GetRasterBand(idx_gdal).ReadAsArray().astype(np.float32)
        b05 = ds_b05.GetRasterBand(idx_gdal).ReadAsArray().astype(np.float32)

        inv_b3 = 1 / b03
        inv_b5 = 1 / b05
        nari = (inv_b3 - inv_b5) / (inv_b3 + inv_b5)
        
        for c in [1, 2, 3, 4]:
            mask_class = (arr_samples_2d == c)
            r, c_dim = min(nari.shape[0], mask_class.shape[0]), min(nari.shape[1], mask_class.shape[1])
            vals = nari[:r, :c_dim][mask_class[:r, :c_dim]]
            vals = vals[~np.isnan(vals)]
            
            if len(vals) > 0:
                nari_means[c].append(np.mean(vals))
                nari_stds[c].append(np.std(vals))
            else:
                nari_means[c].append(np.nan)
                nari_stds[c].append(np.nan)

    ds_b03 = ds_b05 = None

    plt.figure(figsize=(12, 6))
    colors = {1: 'brown', 2: 'green', 3: 'purple', 4: 'darkgreen'}
    labels = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbres'}
    x_axis = range(len(dates))

    for c in [1, 2, 3, 4]:
        mu = np.array(nari_means[c])
        sigma = np.array(nari_stds[c])
        if not np.all(np.isnan(mu)):
            plt.plot(x_axis, mu, label=labels[c], color=colors[c], marker='o')
            plt.fill_between(x_axis, mu - sigma, mu + sigma, color=colors[c], alpha=0.2)

    plt.xticks(x_axis, dates, rotation=45)
    plt.title("Évolution temporelle du NARI par strate (Pyrénées 24-25)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path)
    plt.show()

def create_nari_raster(base_path, output_path):
    """
    Crée un raster multi-bandes de l'indice NARI.
    """
    path_b03 = os.path.join(base_path, 'pyrenees_24-25_B03.tif')
    path_b05 = os.path.join(base_path, 'pyrenees_24-25_B05.tif')
    nodata_val = -9999.0

    ds_src = gdal.Open(path_b03)
    ds_b05 = gdal.Open(path_b05)
    if ds_src is None or ds_b05 is None:
        raise FileNotFoundError("Bandes B03 ou B05 introuvables.")

    x_size, y_size, nb_bands = ds_src.RasterXSize, ds_src.RasterYSize, ds_src.RasterCount
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(output_path, x_size, y_size, nb_bands, gdal.GDT_Float32)
    ds_out.SetGeoTransform(ds_src.GetGeoTransform())
    ds_out.SetProjection(ds_src.GetProjection())

    for i in range(nb_bands):
        idx = i + 1
        arr_b3 = ds_src.GetRasterBand(idx).ReadAsArray().astype(np.float32)
        arr_b5 = ds_b05.GetRasterBand(idx).ReadAsArray().astype(np.float32)
        nari = np.full(arr_b3.shape, nodata_val, dtype=np.float32)
        valid_mask = (arr_b3 > 0) & (arr_b5 > 0)

        inv_b3, inv_b5 = 1.0 / arr_b3[valid_mask], 1.0 / arr_b5[valid_mask]
        nari[valid_mask] = (inv_b3 - inv_b5) / (inv_b3 + inv_b5)
        nari[np.isnan(nari)] = nodata_val
        
        out_band = ds_out.GetRasterBand(idx)
        out_band.WriteArray(nari)
        out_band.SetNoDataValue(nodata_val)

    ds_src = ds_b05 = ds_out = None
    print(f"Raster NARI créé : {output_path}")
    
# =============================================================================
# ETAPE 3 : MACHINE LEARNING
# =============================================================================

def prepare_classification_data(base_dir, image_ref_path, shp_path, out_raster_samples, band_names, extra_indices=None):    
    """
    Extrait les primitives (X), les labels (Y) et les groupes (IDs polygones).
    """
    out_raster_ids = os.path.join(base_dir, 'PI_ids_rasterized.tif')
    raster_ds = gdal.Open(image_ref_path)
    geotransform, projection = raster_ds.GetGeoTransform(), raster_ds.GetProjection()
    x_size, y_size = raster_ds.RasterXSize, raster_ds.RasterYSize
    raster_ds = None

    driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(out_raster_ids): os.remove(out_raster_ids)
    target_ds = driver.Create(out_raster_ids, x_size, y_size, 1, gdal.GDT_UInt32)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)

    shp_ds = ogr.Open(shp_path)
    layer = shp_ds.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=id"])
    target_ds = shp_ds = None

    X_list = []
    file_pattern = "pyrenees_24-25_{}.tif"
    for b_name in band_names:
        fname = os.path.join(base_dir, file_pattern.format(b_name))
        X_band, Y, t = cla.get_samples_from_roi(fname, out_raster_samples)
        X_list.append(X_band)
    
    if extra_indices:
        for extra_path in extra_indices:
            X_idx, _, _ = cla.get_samples_from_roi(extra_path, out_raster_samples)
            X_list.append(X_idx)

    X = np.concatenate(X_list, axis=1)
    ref_band_path = os.path.join(base_dir, file_pattern.format(band_names[0]))
    _, groups, _ = cla.get_samples_from_roi(ref_band_path, out_raster_ids)

    Y, groups = np.squeeze(Y), np.squeeze(groups)
    print(f"Données prêtes : X{X.shape}, Y{Y.shape}, Groupes : {len(np.unique(groups))}")
    return X, Y, groups

def optimize_random_forest(X, Y, groups):
    """
    Optimisation des hyperparamètres via GridSearchCV et StratifiedGroupKFold.
    """
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [None, 10, 15, 20],
        'max_features': [None, 'sqrt', 'log2'],
        'min_samples_leaf': [1, 5]
    }
    cv = StratifiedGroupKFold(n_splits=5)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, 
                               scoring='f1_weighted', n_jobs=-1, verbose=1)
    
    print("Lancement de l'optimisation...")
    grid_search.fit(X, Y, groups=groups)
    print(f"Meilleurs paramètres : {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X, Y, groups, target_names, output_fig_path_matrix, output_fig_path_quality, nb_iter=5):
    """
    Évaluation par validation croisée (K-Fold) :
    1. Affiche les matrices brutes de chaque matrice côte à côte.
    2. Affiche la matrice de confusion moyenne.
    3. Affiche la qualité des classes avec écart-type.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    list_cm = []
    list_accuracy = []
    list_report = []

    # 1. Préparation du découpage spatial (respect des polygones)
    kf = StratifiedGroupKFold(n_splits=nb_iter)

    print(f"Lancement de la validation itérative ({nb_iter} matrices)...")
    
    for i, (train_idx, test_idx) in enumerate(kf.split(X, Y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Entraînement et prédiction
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        # Calcul et stockage des résultats des matrices
        cm_fold = confusion_matrix(Y_test, Y_pred)
        list_cm.append(cm_fold)
        list_accuracy.append(accuracy_score(Y_test, Y_pred))
        
        report = classification_report(Y_test, Y_pred, target_names=target_names, output_dict=True)
        df_temp = pd.DataFrame(report).transpose().iloc[:-3, :] 
        list_report.append(df_temp)

    # --- AFFICHAGE 1 : LES MATRICES INDIVIDUELLES CÔTE À CÔTE ---
    fig, axes = plt.subplots(1, nb_iter, figsize=(22, 5))
    for i in range(nb_iter):
        ax = axes[i]
        cm_fold = list_cm[i]
        ax.imshow(cm_fold, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Matrice {i+1} (Pixels bruts)')
        
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names, rotation=45, fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names, fontsize=9)

        # Ajout des valeurs entières
        for row, col in np.ndindex(cm_fold.shape):
            ax.text(col, row, format(cm_fold[row, col], 'd'),
                     ha="center", va="center",
                     color="white" if cm_fold[row, col] > cm_fold.max()/2 else "black")
    plt.tight_layout()
    plt.show()

    # --- CALCULS DES MOYENNES ET ECARTS-TYPES ---
    mean_cm = np.array(list_cm).mean(axis=0)
    array_acc = np.array(list_accuracy)
    mean_acc, std_acc = array_acc.mean(), array_acc.std()

    all_reports = pd.concat(list_report)
    mean_df_report = all_reports.groupby(level=0).mean()
    std_df_report = all_reports.groupby(level=0).std()

    # --- AFFICHAGE 2 : MATRICE MOYENNE (Style image_be6674.jpg) ---
    plt.figure(figsize=(8, 8))
    plt.imshow(mean_cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Matrice de Confusion Moyenne')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    for i, j in np.ndindex(mean_cm.shape):
        plt.text(j, i, format(mean_cm[i, j], '.1f'), ha="center", va="center",
                 color="white" if mean_cm[i, j] > mean_cm.max()/2 else "black")

    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    if output_fig_path_matrix:
        plt.savefig(output_fig_path_matrix, bbox_inches='tight')
    plt.show()

    # --- AFFICHAGE 3 : ESTIMATION QUALITÉ (Style image_bebd2b.jpg) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    
    mean_df_report[metrics_to_plot].plot.bar(ax=ax, yerr=std_df_report[metrics_to_plot], zorder=2, capsize=4)
    
    ax.set_ylim(0.4, 1.05)
    ax.set_facecolor('ivory') # Fond ivoire du TD
    ax.set_title('Estimation de la qualité des classes (Moyenne + Écart-type)')
    
    # OA moyenne ± écart-type
    ax.text(0.05, 0.95, f'OA : {mean_acc:.2f} ± {std_acc:.2f}', transform=ax.transAxes, 
            fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--', linewidth=0.5, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    if output_fig_path_quality:
        plt.savefig(output_fig_path_quality, bbox_inches='tight')
    plt.show()

    return model

def plot_feature_importance(model, band_names, dates, output_path, top_n=15):
    """
    Calcule et affiche les n variables les plus importantes en reconstruisant 
    les noms à partir des bandes et des dates.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # 1. Reconstruction de la liste des noms (ex: B02_2024-10-11)
    feature_names = []
    for b in band_names:
        for d in dates:
            feature_names.append(f"{b}_{d}")
            
    # Ajout du NARI à la fin de ton X, 
    # il faut ajouter les noms NARI à la liste
    if len(model.feature_importances_) > len(feature_names):
        for d in dates:
            feature_names.append(f"NARI_{d}")

    # 2. Calcul des importances
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=feature_names)
    
    # 'top_n' meilleures variables
    top_features = feat_importances.nlargest(top_n).sort_values(ascending=True)

    # 3. Affichage style 
    plt.figure(figsize=(12, 8))
    top_features.plot(kind='barh', color='tab:blue')
    plt.title(f"Top {top_n} des variables les plus discriminantes")
    plt.xlabel("Importance (Gini)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    print(f"Graphique d'importance sauvegardé : {output_path}")

def plot_data_analysis(X, Y, fig_dir):
    """
    Analyses spectrales saisonnières regroupées par 3 : 
    Histo NIR | Scatter 2D (R-NIR) | Scatter 3D (R-G-NIR)
    """
    import os
    os.makedirs(fig_dir, exist_ok=True)
    
    colors = {1: 'brown', 2: 'yellow', 3: 'purple', 4: 'darkgreen'}
    labels = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbres'}

    # Dictionnaire des indices de colonnes (B03, B04, B08) pour 4 dates clés
    saisons = {
        'Automne (21 Oct)':   {'idx_v': 11, 'idx_r': 21, 'idx_n': 61},
        'Hiver (14 Jan)':     {'idx_v': 14, 'idx_r': 24, 'idx_n': 64},
        'Printemps (29 Mai)': {'idx_v': 16, 'idx_r': 26, 'idx_n': 66},
        'Eté (16 Juil)':      {'idx_v': 18, 'idx_r': 28, 'idx_n': 68}
    }

    for nom_saison, idx in saisons.items():
        s_suffix = nom_saison.split(' ')[0].lower()
        
        # Création d'une figure à 3 colonnes
        fig = plt.figure(figsize=(22, 6))
        fig.suptitle(f"Analyse Spectrale : {nom_saison}", fontsize=16, fontweight='bold')

        # --- 1. HISTOGRAMME NIR (B08) ---
        ax1 = fig.add_subplot(131)
        for c in [1, 2, 3, 4]:
            mask = (Y == c)
            ax1.hist(X[mask, idx['idx_n']], bins=30, alpha=0.5, 
                     label=labels[c], color=colors[c], density=True)
        ax1.set_title(f"Distribution NIR (B08)")
        ax1.set_xlabel("Réflectance")
        ax1.set_ylabel("Densité")
        ax1.legend(fontsize=8)

        # --- 2. SCATTER 2D (Rouge B04 vs NIR B08) ---
        ax2 = fig.add_subplot(132)
        for c in [1, 2, 3, 4]:
            mask = (Y == c)
            ax2.scatter(X[mask, idx['idx_r']], X[mask, idx['idx_n']], 
                        label=labels[c], color=colors[c], alpha=0.5, s=15)
        ax2.set_title(f"Espace Rouge vs NIR")
        ax2.set_xlabel("Rouge (B04)")
        ax2.set_ylabel("NIR (B08)")

        # --- 3. SCATTER 3D (Rouge, Vert, NIR) ---
        ax3 = fig.add_subplot(133, projection='3d')
        for c in [1, 2, 3, 4]:
            mask = (Y == c)
            ax3.scatter(X[mask, idx['idx_r']], X[mask, idx['idx_v']], X[mask, idx['idx_n']], 
                        c=colors[c], label=labels[c], s=20, alpha=0.4)
        ax3.set_title(f"Espace 3D (R-V-NIR)")
        ax3.set_xlabel('B04 (R)')
        ax3.set_ylabel('B03 (V)')
        ax3.set_zlabel('B08 (NIR)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustement pour le titre global
        
        # Sauvegarde d'une seule image par saison contenant les 3 graphes
        save_path = os.path.join(fig_dir, f"analyse_complete_{s_suffix}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    print(f"Analyses saisonnières sauvegardées dans : {fig_dir}")
    

def plot_seasonal_3d_ari(X, Y, fig_dir):
    """
    Génère un graphique 3D interactif (Plotly) par saison.
    Espace : Rouge (B04), Vert (B03), ARI.
    """
    import pandas as pd
    import plotly.express as px
    import numpy as np

    colors_map = {'1': 'brown', '2': 'yellow', '3': 'purple', '4': 'darkgreen'}
    labels_map = {'1': 'Sol Nu', '2': 'Herbe', '3': 'Landes', '4': 'Arbres'}

    # Indices des bandes pour les 4 dates clés
    saisons = {
        'Automne (21 Oct)':   {'idx_v': 11, 'idx_r': 21, 'idx_re': 31},
        'Hiver (14 Jan)':     {'idx_v': 14, 'idx_r': 24, 'idx_re': 34},
        'Printemps (29 Mai)': {'idx_v': 16, 'idx_r': 26, 'idx_re': 36},
        'Eté (16 Juil)':      {'idx_v': 18, 'idx_r': 28, 'idx_re': 38}
    }

    for nom_saison, idx in saisons.items():
        # --- CALCUL ARI (Anthocyanin Reflectance Index) ---
        
        b03 = X[:, idx['idx_v']].astype(float)
        b05 = X[:, idx['idx_re']].astype(float)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ari = ((1/b03) - (1/b05)) / ((1/b03) + (1/b05))
            ari = np.nan_to_num(ari)

        # --- PREPARATION DU DATAFRAME POUR PLOTLY ---
        df = pd.DataFrame({
            'Rouge': X[:, idx['idx_r']],
            'Vert': X[:, idx['idx_v']],
            'ARI': ari,
            'Classe': Y.astype(int).astype(str)
        })
        
        # Mapping des noms de classes pour le survol (hover)
        df['Nom'] = df['Classe'].map(labels_map)

        # --- CREATION DU PLOT 3D INTERACTIF ---
        fig = px.scatter_3d(
            df, x="Rouge", y="Vert", z="ARI",
            color="Classe",
            color_discrete_map=colors_map,
            labels={'Rouge': 'B04 (R)', 'Vert': 'B03 (V)', 'ARI': 'Indice ARI'},
            title=f"Espace Spectral Interactif : {nom_saison}",
            opacity=0.7,
            hover_data=['Nom'],
            height=800, width=1000
        )

        # Réglage de la taille des points pour plus de lisibilité
        fig.update_traces(marker=dict(size=3))

        # Affichage direct dans le notebook (permet la rotation)
        fig.show()

def produce_final_map(model, base_dir, band_names, output_path):
    """
    Génère la carte finale 
    """
    import os
    import numpy as np
    from osgeo import gdal
    
    print("--- Production de la Carte Finale ---")
    file_pattern = "pyrenees_24-25_{}.tif"
    nari_file = 'results/ARI_serie_temp.tif' 
    X_full_list = []
    ref_ds = None 

    # 1. Chargement des bandes
    for b_name in band_names:
        fname = os.path.join(base_dir, file_pattern.format(b_name))
        if ref_ds is None:
            ref_ds = rw.open_image(fname)
            rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize
        img_arr = rw.load_img_as_array(fname)
        X_full_list.append(img_arr.reshape(-1, img_arr.shape[2]))

    # 2. Ajout du NARI
    nari_arr = rw.load_img_as_array(nari_file)
    X_full_list.append(nari_arr.reshape(-1, nari_arr.shape[2]))

    # 3. Création de la matrice X
    X_full = np.nan_to_num(np.concatenate(X_full_list, axis=1))
    
    # --- LA CORRECTION EST ICI ---
    # On crée un masque : un pixel est valide seulement s'il n'est pas tout à 0
    # On vérifie sur les premières bandes spectrales (B02)
    mask_valid = np.all(X_full[:, :10] > 0, axis=1) 
    
    # Initialisation de la carte à 0 (NoData) partout
    Y_full_pred = np.zeros(X_full.shape[0], dtype=np.uint8)

    print(f"Prédiction sur {np.sum(mask_valid)} pixels valides...")
    if np.sum(mask_valid) > 0:
        # On ne prédit QUE les pixels à l'intérieur du masque
        Y_full_pred[mask_valid] = model.predict(X_full[mask_valid])

    # 4. Reconstruction et Sauvegarde
    map_pred = Y_full_pred.reshape(rows, cols)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rw.write_image(output_path, map_pred.astype(np.uint8), data_set=ref_ds, gdal_dtype=gdal.GDT_Byte)
    
    # On définit officiellement le 0 comme NoData pour QGIS
    ds_final = gdal.Open(output_path, gdal.GA_Update)
    if ds_final:
        ds_final.GetRasterBand(1).SetNoDataValue(0)
        ds_final = None
    print(f"✅ Carte terminée : {output_path}")