import json, sys

with open('Codigo-TFM_Ariel_Mery_Ibarra.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def mk_md(lines):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': lines}

def mk_code(lines):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': lines}

# ── Cell [60]: Etapa 5 intro (replace existing) ──────────────────────────────
nb['cells'][60] = mk_md([
    '---\n',
    '## ETAPA 5 — Ciclo de Refinamiento: Balanceo de Clases con SMOTE\n',
    '\n',
    'La quinta etapa del proceso experimental aborda la problemática del **desbalance de clases** '
    'mediante la técnica **SMOTE** (*Synthetic Minority Over-sampling Technique*, Chawla et al., 2002). '
    'A diferencia de la estrategia `class_weight=\'balanced\'` empleada en etapas anteriores —que ajusta '
    'los pesos de las muestras existentes durante el entrenamiento—, SMOTE genera **instancias '
    'sintéticas** de la clase minoritaria interpolando en el espacio de características entre muestras '
    'reales de fraude y sus vecinos más cercanos (*k*-NN con *k* = 5 por defecto).\n',
    '\n',
    'El objetivo de esta etapa es evaluar si la generación de datos sintéticos balanceados permite al '
    '**modelo ganador LightGBM** (con los hiperparámetros óptimos identificados en la Etapa 4) alcanzar '
    'un mayor **recall** sin incurrir en una pérdida significativa de precisión. SMOTE se aplica '
    '**exclusivamente sobre el conjunto de entrenamiento** (*X_train*); el conjunto de validación '
    '(*X_val*) permanece inalterado en su distribución original para garantizar una evaluación '
    'realista.\n',
    '\n',
    'El proceso comprende tres pasos: (1) aplicación de SMOTE sobre *X_train*, '
    '(2) reentrenamiento de LightGBM optimizado sobre los datos balanceados, y '
    '(3) búsqueda del umbral óptimo de clasificación sobre *X_val*.\n',
])

# ── Cell A: SMOTE application ─────────────────────────────────────────────────
cell_smote = mk_code([
    '# ── ETAPA 5.1 — Aplicación de SMOTE ──────────────────────────────────────────\n',
    'from imblearn.over_sampling import SMOTE\n',
    'from collections import Counter\n',
    'import time\n',
    '\n',
    'print("=== ETAPA 5 — SMOTE: Balanceo sintético de clases ===")\n',
    'print()\n',
    '\n',
    'c_before = Counter(y_train)\n',
    'print(f"Distribución ANTES de SMOTE:")\n',
    'print(f"  Clase 0 (Legítima): {c_before[0]:>10,}")\n',
    'print(f"  Clase 1 (Fraude):   {c_before[1]:>10,}")\n',
    'print(f"  Ratio desbalance:   {c_before[0]/c_before[1]:.1f}:1")\n',
    'print()\n',
    '\n',
    '# SMOTE: genera muestras sintéticas de fraude por interpolación k-NN\n',
    '# Puede tardar 20-40 min con 472K filas x 713 features\n',
    't0 = time.time()\n',
    'smote = SMOTE(random_state=SEED, k_neighbors=5)\n',
    'X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)\n',
    'elapsed_sm = time.time() - t0\n',
    '\n',
    'c_after = Counter(y_train_sm)\n',
    'print(f"Distribución DESPUÉS de SMOTE ({elapsed_sm/60:.1f} min):")\n',
    'print(f"  Clase 0 (Legítima): {c_after[0]:>10,}")\n',
    'print(f"  Clase 1 (Fraude):   {c_after[1]:>10,}")\n',
    'print(f"  Ratio desbalance:   {c_after[0]/c_after[1]:.1f}:1")\n',
    'print(f"  Shape X_train_sm:   {X_train_sm.shape}")\n',
])

# ── Cell B: LightGBM retraining on SMOTE data ─────────────────────────────────
cell_lgbm_smote = mk_code([
    '# ── ETAPA 5.2 — Reentrenamiento LightGBM optimizado sobre datos SMOTE ────────\n',
    'from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score\n',
    'import lightgbm as lgb\n',
    '\n',
    '# Recuperar mejores hiperparámetros de Etapa 4\n',
    "best_params_lgbm = resultados_opt['LightGBM']['params'].copy()\n",
    '\n',
    'print("=== LightGBM — Mejores hiperparámetros (Etapa 4) ===")\n',
    'for k, v in best_params_lgbm.items():\n',
    '    print(f"  {k}: {v}")\n',
    'print()\n',
    '\n',
    '# Entrenar sobre datos balanceados con SMOTE\n',
    '# Sin is_unbalance/class_weight: el balance ya está resuelto por SMOTE\n',
    't0 = time.time()\n',
    'modelo_lgbm_smote = lgb.LGBMClassifier(\n',
    '    **best_params_lgbm,\n',
    '    random_state=SEED,\n',
    '    n_jobs=-1,\n',
    '    verbose=-1\n',
    ')\n',
    'modelo_lgbm_smote.fit(X_train_sm, y_train_sm)\n',
    'elapsed_train = time.time() - t0\n',
    'print(f"Entrenamiento completado en {elapsed_train/60:.1f} min")\n',
    '\n',
    '# Predicciones sobre X_val (distribución real — sin SMOTE)\n',
    'y_proba_smote = modelo_lgbm_smote.predict_proba(X_val)[:, 1]\n',
    'auc_smote     = roc_auc_score(y_val, y_proba_smote)\n',
    '\n',
    '# Búsqueda del umbral óptimo (maximiza F1 en X_val)\n',
    'umbrales_e5  = np.arange(0.10, 0.91, 0.01)\n',
    'f1_scores_e5 = [f1_score(y_val, (y_proba_smote >= u).astype(int), zero_division=0)\n',
    '                for u in umbrales_e5]\n',
    'umbral_smote = umbrales_e5[np.argmax(f1_scores_e5)]\n',
    'y_pred_smote = (y_proba_smote >= umbral_smote).astype(int)\n',
    '\n',
    'f1_smote   = f1_score(y_val, y_pred_smote, zero_division=0)\n',
    'prec_smote = precision_score(y_val, y_pred_smote, zero_division=0)\n',
    'rec_smote  = recall_score(y_val, y_pred_smote, zero_division=0)\n',
    '\n',
    'print(f"\\n=== RESULTADOS LightGBM + SMOTE ===")\n',
    'print(f"  AUC:             {auc_smote:.4f}")\n',
    'print(f"  F1 (umbral opt): {f1_smote:.4f}")\n',
    'print(f"  Precisión:       {prec_smote:.4f}")\n',
    'print(f"  Recall:          {rec_smote:.4f}")\n',
    'print(f"  Umbral óptimo:   {umbral_smote:.2f}")\n',
])

# ── Cell C: Comparison table Etapa 4 vs Etapa 5 ───────────────────────────────
cell_comparativa = mk_code([
    '# ── ETAPA 5.3 — Comparativa: LightGBM Etapa 4 vs LightGBM + SMOTE ──────────\n',
    'import pandas as pd\n',
    '\n',
    '# Métricas LightGBM Etapa 4 (recuperadas de resultados_opt)\n',
    "auc_e4  = resultados_opt['LightGBM']['AUC_val']\n",
    "f1_e4   = resultados_opt['LightGBM']['F1_val']\n",
    "prec_e4 = resultados_opt['LightGBM']['Precision_val']\n",
    "rec_e4  = resultados_opt['LightGBM']['Recall_val']\n",
    "umb_e4  = resultados_opt['LightGBM']['Umbral']\n",
    '\n',
    'df_comp5 = pd.DataFrame({\n',
    "    'Etapa'        : ['4 — LightGBM opt.', '5 — LightGBM + SMOTE'],\n",
    "    'AUC'          : [auc_e4,   auc_smote],\n",
    "    'F1'           : [f1_e4,    f1_smote],\n",
    "    'Precisión'    : [prec_e4,  prec_smote],\n",
    "    'Recall'       : [rec_e4,   rec_smote],\n",
    "    'Umbral óptimo': [umb_e4,   umbral_smote],\n",
    '})\n',
    "df_comp5 = df_comp5.set_index('Etapa')\n",
    '\n',
    '# Fila de delta\n',
    'delta5 = df_comp5.iloc[1] - df_comp5.iloc[0]\n',
    "delta5.name = 'Δ (SMOTE − Etapa 4)'\n",
    'df_comp5 = pd.concat([df_comp5, delta5.to_frame().T])\n',
    '\n',
    'print("\\n=== COMPARATIVA ETAPA 4 vs ETAPA 5 ===")\n',
    'display(df_comp5.round(4))\n',
])

# ── Cell D: Figura 7 markdown header ─────────────────────────────────────────
cell_fig7_md = mk_md([
    '### Figura 7. Matriz de confusión — LightGBM optimizado + SMOTE (umbral óptimo)\n',
    '\n',
    'Matriz de confusión del modelo LightGBM entrenado sobre datos balanceados con SMOTE, '
    'evaluado sobre *X_val* en su distribución original.\n',
])

# ── Cell E: Confusion matrix Figura 7 ────────────────────────────────────────
cell_cm7 = mk_code([
    '# ── Figura 7 — Matriz de confusión: LightGBM + SMOTE ────────────────────────\n',
    'from sklearn.metrics import confusion_matrix\n',
    'import matplotlib.pyplot as plt\n',
    '\n',
    'cm_smote = confusion_matrix(y_val, y_pred_smote)\n',
    'total_0  = cm_smote[0].sum()\n',
    'total_1  = cm_smote[1].sum()\n',
    '\n',
    "etiquetas = [['TN', 'FP'], ['FN', 'TP']]\n",
    "colores   = [['#1a5c1a', '#c8e6c9'], ['#c8e6c9', '#1a5c1a']]\n",
    '\n',
    'fig, ax = plt.subplots(figsize=(5, 4))\n',
    '\n',
    'for i in range(2):\n',
    '    total_fila = total_0 if i == 0 else total_1\n',
    '    for j in range(2):\n',
    '        val_abs = cm_smote[i, j]\n',
    '        val_pct = 100 * val_abs / total_fila\n',
    '        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color=colores[i][j]))\n',
    '        linea1 = etiquetas[i][j]\n',
    '        linea2 = f"{val_abs:,}"\n',
    '        linea3 = f"({val_pct:.1f}%)"\n',
    '        texto  = linea1 + "\\n" + linea2 + "\\n" + linea3\n',
    "        color_txt = 'white' if colores[i][j] == '#1a5c1a' else 'black'\n",
    '        ax.text(j, i, texto, ha=\'center\', va=\'center\',\n',
    '                fontsize=12, fontweight=\'bold\', color=color_txt)\n',
    '\n',
    'ax.set_xlim(-0.5, 1.5)\n',
    'ax.set_ylim(-0.5, 1.5)\n',
    'ax.set_xticks([0, 1])\n',
    "ax.set_xticklabels(['Legítima (0)', 'Fraude (1)'])\n",
    'ax.set_yticks([0, 1])\n',
    "ax.set_yticklabels(['Legítima (0)', 'Fraude (1)'])\n",
    "ax.set_xlabel('Predicción')\n",
    "ax.set_ylabel('Real')\n",
    'ax.set_title(f\'LightGBM + SMOTE\\n(umbral = {umbral_smote:.2f})\', fontweight=\'bold\')\n',
    'ax.invert_yaxis()\n',
    '\n',
    'plt.tight_layout()\n',
    "fig_path = 'fig_07_confusion_matrix_smote.png'\n",
    'plt.savefig(fig_path, dpi=150, bbox_inches=\'tight\')\n',
    'plt.show()\n',
    'print(f"Figura 7 guardada en {fig_path}")\n',
])

# ── Cell F: Interpretation placeholder ───────────────────────────────────────
cell_interp5 = mk_md([
    '#### Resultados — Ciclo de Refinamiento: SMOTE sobre LightGBM optimizado\n',
    '\n',
    '*[Esta sección se completará con los resultados reales tras ejecutar las celdas anteriores. '
    'Comparte la tabla comparativa y la Figura 7 para insertar la interpretación con los valores exactos.]*\n',
])

# ── Insertar celdas nuevas en posición 61 (antes de Etapa 6) ─────────────────
new_cells = [cell_smote, cell_lgbm_smote, cell_comparativa, cell_fig7_md, cell_cm7, cell_interp5]
nb['cells'] = nb['cells'][:61] + new_cells + nb['cells'][61:]

with open('Codigo-TFM_Ariel_Mery_Ibarra.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Notebook guardado. Total celdas: {len(nb["cells"])}')
print('Nuevas celdas insertadas en posiciones 61-66')
print('Etapa 6 queda en celdas 67-68')
