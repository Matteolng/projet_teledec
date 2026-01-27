import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal, ogr

# Scikit-Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Librairies internes
from libsigma import read_and_write as rw
from libsigma import classification as cla

# Constantes Globales
COLORS = {1: 'brown', 2: 'yellow', 3: 'purple', 4: 'darkgreen'}
LABELS = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbres'}

# =============================================================================
# 1. PREPARATION ET I/O
# =============================================================================

def rasterize_shapefile(ref_path, shp_path, out_path, col='strate'):
    """Rasterisation du shapefile """
    ref = gdal.Open(ref_path)
    if os.path.exists(out_path): os.remove(out_path)
    
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(out_path, ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(ref.GetGeoTransform())
    ds.SetProjection(ref.GetProjection())
    

    shp_ds = ogr.Open(shp_path)
    if shp_ds is None:
        print(f"Erreur : Impossible d'ouvrir le shapefile {shp_path}")
        return
    lyr = shp_ds.GetLayer()
    gdal.RasterizeLayer(ds, [1], lyr, options=[f"ATTRIBUTE={col}", "ALL_TOUCHED=TRUE"])    
    ds = None; shp_ds = None
    print(f"Rasterisation OK : {out_path}")

def plot_poly_counts(shp_path, col, out_path):
    """Histo polygones (Sauvegarde dans out_path)."""
    df = gpd.read_file(shp_path)
    vc = df[col].value_counts().sort_index()
    colors = [COLORS.get(i, 'gray') for i in vc.index]

    plt.figure(figsize=(8, 5))
    plt.bar(vc.index.astype(str), vc.values, color=colors)
    plt.title("Nombre de polygones par strate")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Création du dossier
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.show()
    plt.close() 
    print(f"Figure sauvegardée : {out_path}")

def plot_pixel_counts(img_path, out_path):
    """Histo pixels (Sauvegarde dans out_path)."""
    arr = rw.load_img_as_array(img_path)
    ids, cnt = np.unique(arr[arr != 0], return_counts=True)
    colors = [COLORS.get(i, 'gray') for i in ids]

    plt.figure(figsize=(8, 5))
    plt.bar(ids.astype(str), cnt, color=colors)
    plt.title("Nombre de pixels par strate")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.show()
    plt.close()
    print(f"Figure sauvegardée : {out_path}")

# =============================================================================
# 2. PHENOLOGIE (NARI)
# =============================================================================

def process_nari_phenology(base, dates, mask_path, out_fig):
    """Courbes NARI avec écarts-types."""
    
    mask = rw.load_img_as_array(mask_path)
    if mask.ndim == 3: mask = mask[:,:,0]
    
    res = {c: {'m': [], 's': []} for c in COLORS}
    ds3 = rw.open_image(os.path.join(base, 'pyrenees_24-25_B03.tif'))
    ds5 = rw.open_image(os.path.join(base, 'pyrenees_24-25_B05.tif'))

    for i in range(len(dates)):
        b3 = ds3.GetRasterBand(i+1).ReadAsArray().astype(float)
        b5 = ds5.GetRasterBand(i+1).ReadAsArray().astype(float)
        
        with np.errstate(all='ignore'):
            inv_b3, inv_b5 = 1.0/b3, 1.0/b5
            nari = (inv_b3 - inv_b5) / (inv_b3 + inv_b5)

        for c in COLORS:
            vals = nari[(mask == c) & ~np.isnan(nari)]
            if len(vals) > 0:
                res[c]['m'].append(np.mean(vals))
                res[c]['s'].append(np.std(vals))
            else:
                res[c]['m'].append(np.nan); res[c]['s'].append(np.nan)

    plt.figure(figsize=(12, 6))
    x = range(len(dates))
    for c in COLORS:
        m, s = np.array(res[c]['m']), np.array(res[c]['s'])
        plt.plot(x, m, label=LABELS[c], color=COLORS[c], marker='o')
        plt.fill_between(x, m-s, m+s, color=COLORS[c], alpha=0.2)
    
    plt.xticks(x, dates, rotation=45)
    plt.title("Évolution temporelle du NARI par strate")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.savefig(out_fig)
    plt.show()

def create_nari_raster(base, out):
    """Création Raster NARI."""
    s3 = gdal.Open(os.path.join(base, 'pyrenees_24-25_B03.tif'))
    s5 = gdal.Open(os.path.join(base, 'pyrenees_24-25_B05.tif'))
    
    drv = gdal.GetDriverByName('GTiff')
    out_ds = drv.Create(out, s3.RasterXSize, s3.RasterYSize, s3.RasterCount, gdal.GDT_Float32)
    out_ds.SetGeoTransform(s3.GetGeoTransform()); out_ds.SetProjection(s3.GetProjection())

    for i in range(s3.RasterCount):
        b3 = s3.GetRasterBand(i+1).ReadAsArray().astype(float)
        b5 = s5.GetRasterBand(i+1).ReadAsArray().astype(float)
        
        nari = np.full(b3.shape, -9999.0, dtype='float32')
        valid = (b3 > 0) & (b5 > 0)
        nari[valid] = ((1/b3[valid]) - (1/b5[valid])) / ((1/b3[valid]) + (1/b5[valid]))
        
        bd = out_ds.GetRasterBand(i+1)
        bd.WriteArray(nari); bd.SetNoDataValue(-9999.0)
    print(f"NARI créé : {out}")

# =============================================================================
# 3. CLASSIFICATION & ANALYSE
# =============================================================================

def prepare_classification_data(base, ref, shp, spl_rst, bands, extra=[]):
    """Extraction Données avec gestion des IDs."""
    # On crée le raster d'IDs 
    ids_rst = spl_rst.replace('sample_strata.tif', 'PI_ids_rasterized.tif')
    rasterize_shapefile(ref, shp, ids_rst, 'id')

    # X
    X_parts = []
    for b in bands:
        f = os.path.join(base, f"pyrenees_24-25_{b}.tif")
        x, _, _ = cla.get_samples_from_roi(f, spl_rst)
        X_parts.append(x)
    for f in extra:
        x, _, _ = cla.get_samples_from_roi(f, spl_rst)
        X_parts.append(x)
    X = np.concatenate(X_parts, axis=1)

    # Y et Groupes
    ref_b = os.path.join(base, f"pyrenees_24-25_{bands[0]}.tif")
    _, Y, _ = cla.get_samples_from_roi(ref_b, spl_rst)
    _, G, _ = cla.get_samples_from_roi(ref_b, ids_rst)

    return X, np.squeeze(Y), np.squeeze(G)

def optimize_random_forest(X, Y, G=None):
    """GridSearch """
    rf = RandomForestClassifier(random_state=33)
    p_grid = {'n_estimators': [100, 150, 200], 'max_depth': [None, 15], 'max_features': ['sqrt', 'log2']}
    
    if G is not None:
        cv = StratifiedGroupKFold(5)
        gs = GridSearchCV(rf, p_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
        gs.fit(X, Y, groups=G)
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        gs = GridSearchCV(rf, p_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
        gs.fit(X, Y)
        
    print(f"Best: {gs.best_params_}")
    return gs.best_estimator_

def evaluate_model(model, X, Y, G, names, out_cm, out_qual, nb_iter=5):
    """Affichage """
    l_cm, l_rep, l_acc = [], [], []
    
    if G is not None:
        cv = StratifiedGroupKFold(nb_iter)
        splitter = cv.split(X, Y, groups=G)
    else:
        cv = StratifiedKFold(nb_iter, shuffle=True, random_state=42)
        splitter = cv.split(X, Y)

    for tr, te in splitter:
        model.fit(X[tr], Y[tr])
        pred = model.predict(X[te])
        l_cm.append(confusion_matrix(Y[te], pred))
        l_acc.append(accuracy_score(Y[te], pred))
        d = classification_report(Y[te], pred, target_names=names, output_dict=True)
        l_rep.append(pd.DataFrame(d).T.iloc[:-3])

    # 1. Matrices Côte à Côte
    fig, ax = plt.subplots(1, nb_iter, figsize=(22, 5))
    for i in range(nb_iter):
        im = ax[i].imshow(l_cm[i], interpolation='nearest', cmap=plt.cm.Blues)
        ax[i].set_title(f'Matrice {i+1}')
        ax[i].set_xticks(range(len(names))); ax[i].set_xticklabels(names, rotation=45)
        ax[i].set_yticks(range(len(names))); ax[i].set_yticklabels(names)
        for r, c in np.ndindex(l_cm[i].shape):
            ax[i].text(c, r, f"{l_cm[i][r, c]}", ha="center", va="center",
                       color="white" if l_cm[i][r, c] > l_cm[i].max()/2 else "black")
    plt.tight_layout(); plt.show()

    # 2. Matrice Moyenne
    avg_cm = np.mean(l_cm, axis=0)
    plt.figure(figsize=(8, 8))
    plt.imshow(avg_cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Matrice de Confusion Moyenne')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(names)), names, rotation=45)
    plt.yticks(range(len(names)), names)
    for i, j in np.ndindex(avg_cm.shape):
        plt.text(j, i, f"{avg_cm[i, j]:.1f}", ha="center", va="center",
                 color="white" if avg_cm[i, j] > avg_cm.max()/2 else "black")
    plt.ylabel('Vraie classe'); plt.xlabel('Classe prédite')
    
    if out_cm:
        os.makedirs(os.path.dirname(out_cm), exist_ok=True)
        plt.savefig(out_cm, bbox_inches='tight')
    plt.show()

    # 3. Qualité
    df_all = pd.concat(l_rep)
    mu, std = df_all.groupby(level=0).mean(), df_all.groupby(level=0).std()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    mu[['precision', 'recall', 'f1-score']].plot.bar(ax=ax, yerr=std, zorder=2, capsize=4)
    
    ax.set_ylim(0.4, 1.05)
    ax.set_facecolor('ivory')
    ax.set_title('Estimation de la qualité des classes (Moyenne + Écart-type)')
    ax.text(0.05, 0.95, f'OA : {np.mean(l_acc):.2f} ± {np.std(l_acc):.2f}', transform=ax.transAxes, 
            fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--', linewidth=0.5, zorder=1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    if out_qual:
        os.makedirs(os.path.dirname(out_qual), exist_ok=True)
        plt.savefig(out_qual, bbox_inches='tight')
    plt.show()
    return model

def plot_feature_importance(model, bands, dates, out, top=15):
    """Top 15 features."""
    feats = []
    for b in bands:
        for d in dates: feats.append(f"{b}_{d}")
    if len(model.feature_importances_) > len(feats):
        for d in dates: feats.append(f"NARI_{d}")
    
    s = pd.Series(model.feature_importances_, index=feats).nlargest(top).sort_values()
    plt.figure(figsize=(12, 8))
    s.plot(kind='barh', color='tab:blue')
    plt.title(f"Top {top} des variables les plus discriminantes")
    plt.xlabel("Importance (Gini)"); plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out)
    plt.show()

def plot_data_analysis(X, Y, out_dir):
    """Analyse Saisonnière Complète."""
    os.makedirs(out_dir, exist_ok=True)
    # Indices (Vert, Rouge, NIR)
    saisons = {'Automne (21 Oct)': (11, 21, 61), 'Hiver (14 Jan)': (14, 24, 64), 
               'Printemps (29 Mai)': (16, 26, 66), 'Ete (16 Juil)': (18, 28, 68)}

    for nom, (v, r, n) in saisons.items():
        suffix = nom.split(' ')[0].lower()
        fig = plt.figure(figsize=(22, 6))
        fig.suptitle(f"Analyse Spectrale : {nom}", fontsize=16, fontweight='bold')

        # 1. Histo
        ax1 = fig.add_subplot(131)
        for c in COLORS:
            sub = X[Y == c, n]
            ax1.hist(sub, bins=30, alpha=0.5, label=LABELS[c], color=COLORS[c], density=True)
        ax1.set_title("Distribution NIR (B08)"); ax1.legend(fontsize=8)

        # 2. Scatter 2D
        ax2 = fig.add_subplot(132)
        for c in COLORS:
            m = (Y == c)
            ax2.scatter(X[m, r], X[m, n], label=LABELS[c], color=COLORS[c], alpha=0.5, s=15)
        ax2.set_title("Espace Rouge vs NIR"); ax2.set_xlabel("Rouge (B04)"); ax2.set_ylabel("NIR (B08)")

        # 3. Scatter 3D
        ax3 = fig.add_subplot(133, projection='3d')
        for c in COLORS:
            m = (Y == c)
            ax3.scatter(X[m, r], X[m, v], X[m, n], c=COLORS[c], label=LABELS[c], s=20, alpha=0.4)
        ax3.set_title("Espace 3D (R-V-NIR)")
        ax3.set_xlabel('B04 (R)'); ax3.set_ylabel('B03 (V)'); ax3.set_zlabel('B08 (NIR)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f"analyse_complete_{suffix}.png"))
        plt.show()

def plot_seasonal_3d_ari(X, Y, out_dir):
    """Plotly 3D ARI."""
    # Indices (Vert, Rouge, RedEdge)
    saisons = {'Automne': (11, 21, 31), 'Hiver': (14, 24, 34), 
               'Printemps': (16, 26, 36), 'Ete': (18, 28, 38)}

    for nom, (v, r, re) in saisons.items():
        b3, b5 = X[:, v].astype(float), X[:, re].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ari = np.nan_to_num(((1/b3) - (1/b5)) / ((1/b3) + (1/b5)))
        
        df = pd.DataFrame({'R': X[:, r], 'V': b3, 'ARI': ari, 'Classe': Y.astype(str)})
        df['Nom'] = df['Classe'].astype(int).map(LABELS)
        
        fig = px.scatter_3d(df, x='R', y='V', z='ARI', color='Classe',
                            color_discrete_map={str(k):v for k,v in COLORS.items()},
                            title=f"Espace Spectral Interactif : {nom}", opacity=0.7, hover_data=['Nom'])
        fig.update_traces(marker=dict(size=3)); fig.show()

def produce_final_map(model, base, bands, nari_path, out):
    """Inférence Finale."""
    # 1. Stack
    l_arr = []
    for b in bands:
        arr = rw.load_img_as_array(os.path.join(base, f"pyrenees_24-25_{b}.tif"))
        l_arr.append(arr.reshape(-1, arr.shape[2]))
    
    # MODIFICATION : Utilise le chemin en argument
    nari = rw.load_img_as_array(nari_path)
    l_arr.append(nari.reshape(-1, nari.shape[2]))
    
    X_full = np.nan_to_num(np.concatenate(l_arr, axis=1))
    
    # 2. Masque (Premières bandes non nulles)
    mask = np.all(X_full[:, :10] > 0, axis=1)
    
    # 3. Pred
    res = np.zeros(X_full.shape[0], dtype=np.uint8)
    print(f"Prediction sur {mask.sum()} pixels valides...")
    if mask.sum() > 0:
        res[mask] = model.predict(X_full[mask])

    # 4. Ecriture
    ref = rw.open_image(os.path.join(base, f"pyrenees_24-25_{bands[0]}.tif"))
    rw.write_image(out, res.reshape(ref.RasterYSize, ref.RasterXSize), data_set=ref, gdal_dtype=gdal.GDT_Byte)
    
    # Set NoData = 0
    ds = gdal.Open(out, 1); ds.GetRasterBand(1).SetNoDataValue(0); ds = None
    print(f" Carte terminée : {out}")