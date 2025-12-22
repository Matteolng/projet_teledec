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
    """
    EN ENTRÉE :
        - image_ref_path (str) : Chemin de l'image Sentinel-2 de référence (Pour à copier la résolution et l'emprise).
        - shp_path (str) : Chemin du fichier vecteur .shp contenant les polygones PI.
        - output_raster (str) : Chemin du fichier .tif qui sera créé.
        - attribute_col (str) : Nom de la colonne : 'strate' pour les classes 1, 2, 3, 4).

    ETAPES
        1. Ouvre l'image de référence pour récupérer ses métadonnées géographiques (GeoTransform, Projection, Taille).
        2. Crée un nouveau fichier GeoTIFF vide avec ces mêmes caractéristiques techniques pour assurer un alignement parfait pixel par pixel.
        3. Utilise l'outil 'gdal.RasterizeLayer' pour transformer les polygones en pixels. Chaque pixel situé sous un polygone prendra la valeur numérique de la strate correspondante.

        CE QUE ÇA SORT EN SORTIE :
        - Un fichier raster (.tif)  
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
    EN ENTRÉE :
    - shp_path (str) : Chemin vers le fichier vecteur d'échantillons (.shp).
    - col_classe (str) : Nom de la colonne contenant les labels des classes : strate.
    - output_path (str) : Chemin où enregistrer l'image du graphique 

    EN SORTIE :
    - Un fichier image (.png) dans ton dossier de résultats.
    - Un plot dans le Notebook pour vérifier si les classes sont équilibrées 
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

    EN ENTRÉE :
    - raster_path (str) : Chemin vers le fichier raster des échantillons (.tif) créé par rasterize_shapefile.
    - output_path (str) : Chemin  où enregistrer le graphique 

    EN SORTIE :
    - Un fichier image (.png) montrant la répartition des pixels par strate.

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

def process_nari_phenology(base_path, dates, output_fig_path):

    """

    EN ENTRÉE :
    base_path (str) : Chemin vers le dossier contenant les images Sentinel-2 (bandes B03 et B05).
    dates (list) : Liste des dates correspondant aux bandes de la série temporelle.
    output_fig_path (str) : Chemin où enregistrer le graphique de phénologie.

    EN SORTIE :
    Un fichier image (.png) : Graphique montrant l'évolution moyenne de l'indice NARI (avec rubans d'écart-type) pour les 4 strates étudiées sur l'ensemble des dates.

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

        inv_b3 = 1 / b03
        inv_b5 = 1 / b05
        # Formule NARI 
        nari = (inv_b3 - inv_b5) / (inv_b3 + inv_b5)
        
        for c in [1, 2, 3, 4]:
            mask_class = (arr_samples_2d == c)
            # Gestion dimensions
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

    """

    EN ENTRÉE :
    base_path (str) : Chemin vers le dossier contenant les séries temporelles des bandes B03 (Vert) et B05 (Red Edge).
    output_path (str) : Chemin où enregistrer le nouveau raster produit.

    EN SORTIE :
    Un fichier GeoTIFF (.tif) : Une image multi-bandes au format Float32 contenant les valeurs de l'indice NARI calculées pour chaque date de la série, avec une valeur de NoData fixée à -9999.0.

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
    

def prepare_classification_data(base_dir, image_ref_path, shp_path, out_raster_samples, band_names):
    """
    Rasterise les IDs, construit X, Y et extrait les groupes.
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
    target_ds = driver.Create(out_raster_ids, x_size, y_size, 1, gdal.GDT_UInt32)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)

    shp_ds = ogr.Open(shp_path)
    layer = shp_ds.GetLayer()
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
        X_band = img_arr[pixel_locations[0], pixel_locations[1], :]
        X_list.append(X_band)

    X = np.nan_to_num(np.concatenate(X_list, axis=1))

    # --- 3. EXTRACTION DES GROUPES ---
    arr_ids = rw.load_img_as_array(out_raster_ids)
    if arr_ids.ndim == 3: arr_ids = arr_ids[:,:,0]
    groups = arr_ids[pixel_locations]

    print(f"Données prêtes : X{X.shape}, Y{Y.shape}, Groupes : {len(np.unique(groups))}")
    return X, Y, groups

def optimize_random_forest(X, Y, groups):
    """
    Optimise le modèle sur l'ensemble des données fournies.
    """
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 15, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 5]
    }
    
    cv = StratifiedGroupKFold(n_splits=5)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, 
                               scoring='f1_weighted', n_jobs=-1, verbose=1)
    
    print("Optimisation en cours...")
    grid_search.fit(X, Y, groups=groups)
    print(f"Meilleurs paramètres : {grid_search.best_params_}")
    
    return grid_search.best_estimator_
from sklearn.model_selection import train_test_split

def evaluate_model(model, X, Y, target_names, output_fig_path):
    """
    Découpe les données en 70% entraînement / 30% test pour 
    obtenir une évaluation réaliste.
    """
    # 1. On sépare les données (sans GroupShuffleSplit, juste aléatoirement)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.30, random_state=42, stratify=Y
    )

    # 2. On ré-entraîne le modèle sur la partie 'train' uniquement
    model.fit(X_train, Y_train)

    # 3. On prédit sur la partie 'test' (les pixels que le modèle n'a pas vus)
    Y_pred = model.predict(X_test)

    print("\n RAPPORT DE CLASSIFICATION (Sur 30% de test)")
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    
    # 4. Affichage de la matrice
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matrice de Confusion (Données de Test)')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path)
    plt.show()

    return model

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_feature_importance_dates(model, band_names, dates, output_path):
    """
    Génère et sauvegarde le Top 15 de l'importance des variables 

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
        print(f" Warning: {len(noms_vars_reels)} noms générés pour {len(importances)} variables.")
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
    Génère des analyses spectrales (Histo, 2D, 3D) pour chaque saison.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    os.makedirs(fig_dir, exist_ok=True)
    
    # Herbe en jaune (2)
    colors = {1: 'brown', 2: 'yellow', 3: 'purple', 4: 'darkgreen'}
    labels = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbres'}

    # Définition des index (B02: 0-9, B03: 10-19, B04: 20-29, B08: 60-69)
    saisons = {
        'Automne (21 Oct)':   {'idx_v': 11, 'idx_r': 21, 'idx_n': 61},
        'Hiver (14 Jan)':     {'idx_v': 14, 'idx_r': 24, 'idx_n': 64},
        'Printemps (29 Mai)': {'idx_v': 16, 'idx_r': 26, 'idx_n': 66},
        'Eté (16 Juil)':      {'idx_v': 18, 'idx_r': 28, 'idx_n': 68}
    }

    for nom_saison, idx in saisons.items():
        print(f"Génération des graphiques pour : {nom_saison}")
        s_suffix = nom_saison.split(' ')[0].lower()
        
        # --- 1. HISTOGRAMME (NIR / B08) ---
        plt.figure(figsize=(10, 6))
        for c in [1, 2, 3, 4]:
            mask = (Y == c)
            plt.hist(X[mask, idx['idx_n']], bins=30, alpha=0.5, 
                     label=labels[c], color=colors[c], density=True)
        plt.title(f"Distribution NIR (B08) - {nom_saison}")
        plt.xlabel("Valeur de réflectance")
        plt.legend()
        plt.savefig(os.path.join(fig_dir, f"hist_NIR_{s_suffix}.png"))
        plt.show()

        # --- 2. SCATTER 2D (Rouge vs NIR) ---
        plt.figure(figsize=(8, 8))
        for c in [1, 2, 3, 4]:
            mask = (Y == c)
            plt.scatter(X[mask, idx['idx_r']], X[mask, idx['idx_n']], 
                        label=labels[c], color=colors[c], alpha=0.5, s=15)
        plt.title(f"Espace Rouge vs NIR - {nom_saison}")
        plt.xlabel("Rouge (B04)")
        plt.ylabel("NIR (B08)")
        plt.legend()
        plt.savefig(os.path.join(fig_dir, f"scatter_2D_{s_suffix}.png"))
        plt.show()

        # --- 3. SCATTER 3D (Rouge, Vert, NIR) ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for c in [1, 2, 3, 4]:
            mask = (Y == c)
            # Correction ici : on utilise bien idx['idx_n']
            ax.scatter(X[mask, idx['idx_r']], X[mask, idx['idx_v']], X[mask, idx['idx_n']], 
                       c=colors[c], label=labels[c], s=20, alpha=0.5)
        
        ax.set_xlabel('Rouge (B04)')
        ax.set_ylabel('Vert (B03)')
        ax.set_zlabel('NIR (B08)')
        plt.title(f"Espace 3D - {nom_saison}")
        plt.legend()
        plt.savefig(os.path.join(fig_dir, f"scatter_3D_{s_suffix}.png"))
        plt.show()
    
def produce_final_map(model, base_dir, band_names, output_path):
    """
    Prédit les strates sur l'ensemble de l'image en ignorant le fond (NoData).
    """
    import numpy as np
    import os
    from osgeo import gdal
    from libsigma import read_and_write as rw

    print("--- Production de la Carte Finale ---")
    file_pattern = "pyrenees_24-25_{}.tif"
    
    X_full_list = []
    ref_ds = None 

    # 1. Reconstruction de la matrice de données complète
    for b_name in band_names:
        fname = os.path.join(base_dir, file_pattern.format(b_name))
        
        if ref_ds is None:
            ref_ds = rw.open_image(fname)
            rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize
            print(f"Dimensions : {cols} x {rows}")

        img_arr = rw.load_img_as_array(fname)
        # On aplatit les dimensions spatiales (Pixels, Dates)
        X_full_list.append(img_arr.reshape(-1, img_arr.shape[2]))

    # C'est ici que X_full est définie
    X_full = np.nan_to_num(np.concatenate(X_full_list, axis=1))
    print(f"Données prêtes : {X_full.shape}")

    # 2. Création du masque pour éviter le fond bleu (NoData)
    # On considère qu'un pixel est du "fond" si toutes ses valeurs spectrales sont à 0
    mask_valid = np.any(X_full != 0, axis=1) 
    
    # On initialise la sortie avec des 0 (Classe 0 = NoData/Transparent)
    Y_full_pred = np.zeros(X_full.shape[0], dtype=np.uint8)

    # 3. Prédiction uniquement sur les zones valides
    print(f"Lancement de la prédiction sur {np.sum(mask_valid)} pixels...")
    if np.sum(mask_valid) > 0:
        Y_full_pred[mask_valid] = model.predict(X_full[mask_valid])

    # 4. Remise en forme spatiale
    map_pred = Y_full_pred.reshape(rows, cols)

    # 5. Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rw.write_image(output_path, map_pred.astype(np.uint8), data_set=ref_ds, gdal_dtype=gdal.GDT_Byte)
    
    # On définit explicitement le 0 comme valeur de NoData pour les logiciels SIG
    ds_final = gdal.Open(output_path, gdal.GA_Update)
    if ds_final:
        ds_final.GetRasterBand(1).SetNoDataValue(0)
        ds_final = None
    
    print(f" Carte terminée et sauvegardée : {output_path}")