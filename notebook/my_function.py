import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from osgeo import gdal, ogr
from libsigma import read_and_write as rw


import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix

from libsigma import read_and_write as rw

def rasterize_shapefile(image_ref_path, shp_path, output_raster, attribute_col='strate'):
    """Rasterise le shapefile en s'alignant sur l'image de référence."""
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
    print(f"✅ Rasterisation terminée : {output_raster}")

def plot_poly_counts(shp_path, col_classe, output_path):
    """Génère le diagramme du nombre de polygones."""
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
    """Génère le diagramme du nombre de pixels."""
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

def process_nari_phenology(base_path, dates, output_fig_path):
    """
    Calcule le NARI et affiche la phénologie des strates.
    """
    # Chemins
    path_b03 = os.path.join(base_path, 'pyrenees_24-25_B03.tif')
    path_b05 = os.path.join(base_path, 'pyrenees_24-25_B05.tif')
    output_raster = os.path.join(base_path, 'sample_strata.tif')

    # Initialisation
    nari_means = {1: [], 2: [], 3: [], 4: []}
    nari_stds = {1: [], 2: [], 3: [], 4: []}

    # Chargement masque
    arr_samples = rw.load_img_as_array(output_raster)
    arr_samples_2d = arr_samples[:, :, 0] if arr_samples.ndim == 3 else arr_samples

    # Lecture images
    ds_b03 = rw.open_image(path_b03)
    ds_b05 = rw.open_image(path_b05)

    if ds_b03 is None or ds_b05 is None:
        raise FileNotFoundError("Fichiers B03 ou B05 introuvables.")

    nb_bandes = ds_b03.RasterCount
    
    # Calcul
    for i in range(min(len(dates), nb_bandes)):
        idx_gdal = i + 1
        b03 = ds_b03.GetRasterBand(idx_gdal).ReadAsArray().astype(np.float32)
        b05 = ds_b05.GetRasterBand(idx_gdal).ReadAsArray().astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_b3 = 1 / b03
            inv_b5 = 1 / b05
            # Formule NARI du projet
            nari = (inv_b3 - inv_b5) / (inv_b3 + inv_b5)
        
        for c in [1, 2, 3, 4]:
            mask_class = (arr_samples_2d == c)
            # Gestion sécurité dimensions
            r, c_dim = min(nari.shape[0], mask_class.shape[0]), min(nari.shape[1], mask_class.shape[1])
            vals = nari[:r, :c_dim][mask_class[:r, :c_dim]]
            vals = vals[~np.isnan(vals)]
            
            if len(vals) > 0:
                nari_means[c].append(np.mean(vals))
                nari_stds[c].append(np.std(vals))
            else:
                nari_means[c].append(np.nan)
                nari_stds[c].append(np.nan)

    ds_b03 = None
    ds_b05 = None

    # Graphique
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
    
    # Sauvegarde
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path)
    plt.show()
    


def create_nari_raster(base_path, output_path):
    """Calcule la série temporelle NARI et l'enregistre en GeoTIFF."""
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

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_b3, inv_b5 = 1.0 / arr_b3[valid_mask], 1.0 / arr_b5[valid_mask]
            nari[valid_mask] = (inv_b3 - inv_b5) / (inv_b3 + inv_b5)

        nari[np.isnan(nari)] = nodata_val
        out_band = ds_out.GetRasterBand(idx)
        out_band.WriteArray(nari)
        out_band.SetNoDataValue(nodata_val)

    ds_src = ds_b05 = ds_out = None
    print(f"✅ Raster NARI créé : {output_path}")
    

def prepare_classification_data(base_dir, image_ref_path, shp_path, out_raster_samples, band_names):
    """
    1. Rasterise les IDs des polygones pour créer les groupes.
    2. Construit la matrice X (features) et le vecteur Y (labels).
    3. Extrait les groupes pour la validation croisée groupée.
    """
    # --- 1. RASTERISATION DES IDS ---
    out_raster_ids = os.path.join(base_dir, 'PI_ids_rasterized.tif')
    
    raster_ds = gdal.Open(image_ref_path)
    geotransform = raster_ds.GetGeoTransform()
    projection = raster_ds.GetProjection()
    x_size, y_size = raster_ds.RasterXSize, raster_ds.RasterYSize
    raster_ds = None

    driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(out_raster_ids): os.remove(out_raster_ids)
    target_ds = driver.Create(out_raster_ids, x_size, y_size, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)

    shp_ds = ogr.Open(shp_path)
    layer = shp_ds.GetLayer()
    # On utilise 'id' (ou 'ID') pour différencier chaque polygone
    gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=id"])
    target_ds = None
    shp_ds = None

    # --- 2. CONSTRUCTION DE X ET Y ---
    arr_samples = rw.load_img_as_array(out_raster_samples)
    if arr_samples.ndim == 3: arr_samples = arr_samples[:,:,0]
    
    pixel_locations = np.where(arr_samples > 0)
    Y = arr_samples[pixel_locations]

    X_list = []
    file_pattern = "pyrenees_24-25_{}.tif"
    for b_name in band_names:
        fname = os.path.join(base_dir, file_pattern.format(b_name))
        img_arr = rw.load_img_as_array(fname)
        # Extraction des valeurs pour chaque date (3ème dimension)
        X_band = img_arr[pixel_locations[0], pixel_locations[1], :]
        X_list.append(X_band)

    X = np.concatenate(X_list, axis=1)
    # Remplacement des NaNs éventuels par 0
    X = np.nan_to_num(X)

    # --- 3. EXTRACTION DES GROUPES ---
    arr_ids = rw.load_img_as_array(out_raster_ids)
    if arr_ids.ndim == 3: arr_ids = arr_ids[:,:,0]
    groups = arr_ids[pixel_locations]

    print(f"✅ Données prêtes : X{X.shape}, Y{Y.shape}, Groups{groups.shape}")
    return X, Y, groups



def optimize_random_forest(X, Y, groups):
    """
    Réalise l'optimisation des hyperparamètres avec GridSearchCV et GroupKFold.
    """
    rf = RandomForestClassifier(random_state=42)
    
    # Grille officielle du Tableau 4
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [None, 10, 15, 20],
        'max_features': [None, 'sqrt', 'log2'],
        'min_samples_leaf': [1, 5]
    }
    
    # Validation croisée par groupe (polygones)
    cv = StratifiedGroupKFold(n_splits=5)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    print("🚀 Lancement de l'optimisation (GridSearch + GroupKFold)...")
    grid_search.fit(X, Y, groups=groups)
    
    print(f"✅ Meilleurs paramètres : {grid_search.best_params_}")
    print(f"✅ Meilleur score F1 : {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X, Y, target_names, output_fig_path):
    """
    Affiche le rapport de classification et sauvegarde la matrice de confusion.
    """
    Y_pred = model.predict(X)
    print("\n--- RAPPORT DE CLASSIFICATION ---")
    print(classification_report(Y, Y_pred, target_names=target_names))
    
    cm = confusion_matrix(Y, Y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matrice de Confusion (Données complètes)')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path)
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_feature_importance_dates(model, band_names, dates, output_path):
    """
    Génère et sauvegarde le Top 15 de l'importance des variables 
    en utilisant les couples Bande_Date réels.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # 1. Récupération des scores d'importance du modèle
    importances = model.feature_importances_

    # 2. Génération des noms complets (ex: B02_2024-10-11)
    # L'ordre doit correspondre à la façon dont X a été construit
    noms_vars_reels = []
    for band in band_names:
        for date in dates:
            noms_vars_reels.append(f"{band}_{date}")

    # Vérification de sécurité sur la longueur
    if len(noms_vars_reels) != len(importances):
        print(f"⚠️ Warning: {len(noms_vars_reels)} noms générés pour {len(importances)} variables.")
        noms_vars_reels = noms_vars_reels[:len(importances)]

    # 3. Création du DataFrame pour le tri
    df_imp = pd.DataFrame({'Variable': noms_vars_reels, 'Importance': importances})
    df_top15 = df_imp.sort_values('Importance', ascending=False).head(15)

    # 4. Affichage Graphique
    plt.figure(figsize=(12, 8))
    plt.barh(df_top15['Variable'], df_top15['Importance'], color='#2c7bb6')
    plt.gca().invert_yaxis() # Met la plus importante en haut

    plt.title("Top 15 des variables les plus discriminantes (Bande & Date)", fontsize=14)
    plt.xlabel("Importance (Gini)")
    plt.ylabel("Couple Bande - Date")
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 5. Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

    # Affichage textuel pour ton suivi
    print("\nTop des variables pour le rapport :")
    print(df_top15.to_string(index=False))


    import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D

def plot_data_analysis(X, Y, fig_dir):
    """
    Génère l'ensemble des graphiques d'analyse de séparabilité :
    Histogrammes, Scatter 2D, Signatures temporelles, KDE et Scatter 3D.
    """
    os.makedirs(fig_dir, exist_ok=True)
    colors = {1: 'brown', 2: 'green', 3: 'purple', 4: 'darkgreen'}
    labels = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbres'}

    # --- 1. HISTOGRAMME (B08 - NIR en Mai) ---
    idx_col = 66 
    plt.figure(figsize=(10, 6))
    for c in [1, 2, 3, 4]:
        mask = (Y == c)
        plt.hist(X[mask, idx_col], bins=30, alpha=0.5, label=labels[c], color=colors[c], density=True)
    plt.title(f"Distribution des valeurs - B08 (NIR) - 29 Mai 2025")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "histogramme_B08_Mai.png"))
    plt.show()

    # --- 2. SCATTER 2D (Rouge vs NIR) ---
    idx_red, idx_nir = 26, 66
    plt.figure(figsize=(8, 8))
    for c in [1, 2, 3, 4]:
        mask = (Y == c)
        plt.scatter(X[mask, idx_red], X[mask, idx_nir], label=labels[c], color=colors[c], alpha=0.6, s=15)
    plt.title("Espace spectral : Rouge vs NIR (29 Mai 2025)")
    plt.xlabel("Rouge (B04)")
    plt.ylabel("Proche Infrarouge (B08)")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "scatter_Red_NIR_Mai.png"))
    plt.show()

    # --- 3. SIGNATURE TEMPORELLE (B08) ---
    start_col, end_col = 60, 70
    dates_labels = ['Oct-11', 'Oct-21', 'Nov-28', 'Dec-05', 'Jan-14', 'Feb-23', 'May-29', 'Jun-18', 'Jul-16', 'Aug-24']
    plt.figure(figsize=(12, 6))
    for c in [1, 2, 3, 4]:
        mask = (Y == c)
        sub_X = X[mask, start_col:end_col]
        means, stds = np.mean(sub_X, axis=0), np.std(sub_X, axis=0)
        plt.plot(range(10), means, label=labels[c], color=colors[c], marker='o')
        plt.fill_between(range(10), means - stds, means + stds, color=colors[c], alpha=0.2)
    plt.xticks(range(10), dates_labels, rotation=45)
    plt.title("Signature Temporelle Moyenne (Bande B08 - NIR)")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "signature_temporelle_B08.png"))
    plt.show()

    # --- 4. KDE (DENSITÉ Red-Edge B05) ---
    idx_col_kde = 35
    x_plot = np.linspace(X[:, idx_col_kde].min(), X[:, idx_col_kde].max(), 200)[:, np.newaxis]
    plt.figure(figsize=(10, 6))
    for c in [1, 2, 3, 4]:
        mask = (Y == c)
        vals = X[mask, idx_col_kde]
        if len(vals) > 1:
            kde = KernelDensity(bandwidth=50).fit(vals.reshape(-1, 1))
            density = np.exp(kde.score_samples(x_plot))
            plt.plot(x_plot, density, color=colors[c], lw=2, label=labels[c])
            plt.fill_between(x_plot[:, 0], 0, density, alpha=0.3, color=colors[c])
    plt.title("Densité de probabilité (KDE) - B05 (Red-Edge)")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "kde_B05_Mai.png"))
    plt.show()

    # --- 5. SCATTER 3D ---
    idx_v, idx_r, idx_n = 15, 25, 65
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for c in [1, 2, 3, 4]:
        mask = (Y == c)
        ax.scatter(X[mask, idx_r], X[mask, idx_v], X[mask, idx_n], c=colors[c], label=labels[c], s=20, alpha=0.6)
    ax.set_xlabel('Rouge (B04)')
    ax.set_ylabel('Vert (B03)')
    ax.set_zlabel('NIR (B08)')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "scatter_3d_RGB_NIR.png"))
    plt.show()

def produce_final_map(model, base_dir, band_names, output_path):
    """
    Prédit les strates sur l'ensemble de l'image et sauvegarde le résultat en .tif.
    """
    import numpy as np
    import os
    from osgeo import gdal
    from libsigma import read_and_write as rw

    print("--- Production de la Carte Finale ---")
    file_pattern = "pyrenees_24-25_{}.tif"
    
    X_full_list = []
    ref_ds = None 

    # 1. Construction de la matrice de données complète
    for b_name in band_names:
        fname = os.path.join(base_dir, file_pattern.format(b_name))
        
        if ref_ds is None:
            ref_ds = rw.open_image(fname)
            rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize
            print(f"Dimensions : {cols} x {rows}")

        img_arr = rw.load_img_as_array(fname)
        # On aplatit les dimensions spatiales (H*W, Dates)
        X_full_list.append(img_arr.reshape(-1, img_arr.shape[2]))

    X_full = np.nan_to_num(np.concatenate(X_full_list, axis=1))
    print(f"Données prêtes : {X_full.shape}")

    # 2. Prédiction
    print("Lancement de la prédiction...")
    Y_full_pred = model.predict(X_full)

    # 3. Remise en forme spatiale
    map_pred = Y_full_pred.reshape(rows, cols)

    # 4. Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # On force GDT_Byte car les classes sont des entiers (1 à 4)
    rw.write_image(output_path, map_pred.astype(np.uint8), data_set=ref_ds, gdal_dtype=gdal.GDT_Byte)
    
    print(f"✅ Carte terminée et sauvegardée dans : {output_path}")