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

def prepare_classification_data(base_dir, image_ref_path, shp_path, out_raster_samples, band_names):
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

def evaluate_model(model, X, Y, target_names, output_fig_path):
    """
    Évaluation du modèle par split 70/30 et affichage Matplotlib.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.30, random_state=42, stratify=Y
    )
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    print("\nRAPPORT DE CLASSIFICATION (Test 30%)")
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    
    cm = confusion_matrix(Y_test, Y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Matrice de Confusion (Données de Test)')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(target_names, rotation=45)
    ax.set_yticks(tick_marks); ax.set_yticklabels(target_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('Vraie classe'); ax.set_xlabel('Classe prédite')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, bbox_inches='tight')
    plt.show()
    return model

def plot_feature_importance_dates(model, band_names, dates, output_path):
    """
    Visualisation de l'importance des variables par couple Bande-Date.
    """
    importances = model.feature_importances_
    noms_vars_reels = [f"{band}_{date}" for band in band_names for date in dates]

    if len(noms_vars_reels) != len(importances):
        noms_vars_reels = noms_vars_reels[:len(importances)]

    df_imp = pd.DataFrame({'Variable': noms_vars_reels, 'Importance': importances})
    df_top15 = df_imp.sort_values('Importance', ascending=False).head(15)

    plt.figure(figsize=(12, 8))
    plt.barh(df_top15['Variable'], df_top15['Importance'], color='#2c7bb6')
    plt.gca().invert_yaxis()
    plt.title("Top 15 des variables les plus discriminantes")
    plt.xlabel("Importance (Gini)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def plot_data_analysis(X, Y, fig_dir):
    """
    Analyses spectrales saisonnières (Histogrammes, Scatter 2D et 3D).
    """
    os.makedirs(fig_dir, exist_ok=True)
    colors = {1: 'brown', 2: 'yellow', 3: 'purple', 4: 'darkgreen'}
    labels = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbres'}

    saisons = {
        'Automne (21 Oct)':   {'idx_v': 11, 'idx_r': 21, 'idx_n': 61},
        'Hiver (14 Jan)':     {'idx_v': 14, 'idx_r': 24, 'idx_n': 64},
        'Printemps (29 Mai)': {'idx_v': 16, 'idx_r': 26, 'idx_n': 66},
        'Eté (16 Juil)':      {'idx_v': 18, 'idx_r': 28, 'idx_n': 68}
    }

    for nom_saison, idx in saisons.items():
        s_suffix = nom_saison.split(' ')[0].lower()
        
        # Histo NIR
        plt.figure(figsize=(10, 6))
        for c in [1, 2, 3, 4]:
            mask = (Y == c)
            plt.hist(X[mask, idx['idx_n']], bins=30, alpha=0.5, label=labels[c], color=colors[c], density=True)
        plt.title(f"Distribution NIR (B08) - {nom_saison}")
        plt.legend(); plt.savefig(os.path.join(fig_dir, f"hist_NIR_{s_suffix}.png")); plt.show()

        # Scatter 2D
        plt.figure(figsize=(8, 8))
        for c in [1, 2, 3, 4]:
            mask = (Y == c)
            plt.scatter(X[mask, idx['idx_r']], X[mask, idx['idx_n']], label=labels[c], color=colors[c], alpha=0.5, s=15)
        plt.title(f"Espace Rouge vs NIR - {nom_saison}")
        plt.xlabel("Rouge (B04)"); plt.ylabel("NIR (B08)")
        plt.legend(); plt.savefig(os.path.join(fig_dir, f"scatter_2D_{s_suffix}.png")); plt.show()

        # Scatter 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for c in [1, 2, 3, 4]:
            mask = (Y == c)
            ax.scatter(X[mask, idx['idx_r']], X[mask, idx['idx_v']], X[mask, idx['idx_n']], c=colors[c], label=labels[c], s=20, alpha=0.5)
        ax.set_xlabel('Rouge (B04)'); ax.set_ylabel('Vert (B03)'); ax.set_zlabel('NIR (B08)')
        plt.title(f"Espace 3D - {nom_saison}")
        plt.legend(); plt.savefig(os.path.join(fig_dir, f"scatter_3D_{s_suffix}.png")); plt.show()

def produce_final_map(model, base_dir, band_names, output_path):
    """
    Génère la carte finale classifiée au format GeoTIFF.
    """
    print("--- Production de la Carte Finale ---")
    file_pattern = "pyrenees_24-25_{}.tif"
    X_full_list = []
    ref_ds = None 

    for b_name in band_names:
        fname = os.path.join(base_dir, file_pattern.format(b_name))
        if ref_ds is None:
            ref_ds = rw.open_image(fname)
            rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize
        img_arr = rw.load_img_as_array(fname)
        X_full_list.append(img_arr.reshape(-1, img_arr.shape[2]))

    X_full = np.nan_to_num(np.concatenate(X_full_list, axis=1))
    mask_valid = np.any(X_full != 0, axis=1) 
    Y_full_pred = np.zeros(X_full.shape[0], dtype=np.uint8)

    print(f"Prédiction sur {np.sum(mask_valid)} pixels...")
    if np.sum(mask_valid) > 0:
        Y_full_pred[mask_valid] = model.predict(X_full[mask_valid])

    map_pred = Y_full_pred.reshape(rows, cols)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rw.write_image(output_path, map_pred.astype(np.uint8), data_set=ref_ds, gdal_dtype=gdal.GDT_Byte)
    
    ds_final = gdal.Open(output_path, gdal.GA_Update)
    if ds_final:
        ds_final.GetRasterBand(1).SetNoDataValue(0)
        ds_final = None
    print(f"Carte terminée et sauvegardée : {output_path}")