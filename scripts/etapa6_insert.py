import json, sys

with open('Codigo-TFM_Ariel_Mery_Ibarra.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def mk_md(lines):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': lines}

def mk_code(lines):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': lines}

# ── Cell [67]: Etapa 6 description (replace) ─────────────────────────────────
nb['cells'][67] = mk_md([
    '---\n',
    '## ETAPA 6 — Generación de Predicciones Finales para Kaggle\n',
    '\n',
    'La sexta y última etapa del proceso experimental genera las **predicciones finales sobre el '
    'dataset de test de la competición IEEE-CIS Fraud Detection (Kaggle, 2019)**. El modelo '
    'seleccionado es el **LightGBM optimizado de la Etapa 4** (AUC = 0.9680, F1 = 0.7856, '
    'Precisión = 0.8797, Recall = 0.7097, umbral óptimo = 0.89), que superó al modelo con SMOTE '
    'de la Etapa 5 en F1 y precisión.\n',
    '\n',
    'La competición Kaggle evalúa las predicciones mediante **AUC-ROC**, por lo que el archivo de '
    'entrega (*submission*) contiene la **probabilidad de fraude** por transacción (valor continuo '
    'en [0, 1]) en lugar de etiquetas binarias. El umbral óptimo (0.89) se reserva para uso '
    'operativo, no para la submission de Kaggle.\n',
    '\n',
    'El proceso comprende: (1) imputación de NaN en *X_test_kaggle* con medianas de *X_train*, '
    '(2) generación de probabilidades con el modelo ganador, (3) exportación del archivo '
    '`submission.csv` y (4) análisis de la distribución de probabilidades predichas.\n',
])

# ── Cell [68]: Predictions code (replace) ────────────────────────────────────
nb['cells'][68] = mk_code([
    '# ── ETAPA 6.1 — Preparación del dataset de test ─────────────────────────────\n',
    'import numpy as np\n',
    '\n',
    'print("=== ETAPA 6 — Generación de Predicciones Finales ===")\n',
    'print()\n',
    '\n',
    '# Imputar NaN en X_test_kaggle con medianas de X_train (misma estrategia que Etapa 2)\n',
    'nan_test = X_test_kaggle.isnull().sum().sum()\n',
    'print(f"NaN en X_test_kaggle antes de imputación: {nan_test:,}")\n',
    '\n',
    'medianas_test = X_train.median(numeric_only=True)\n',
    'X_test_final  = X_test_kaggle.fillna(medianas_test)\n',
    '\n',
    'nan_post = X_test_final.isnull().sum().sum()\n',
    'print(f"NaN en X_test_final  tras imputación:     {nan_post:,}")\n',
    'print(f"Shape X_test_final: {X_test_final.shape}")\n',
    'print()\n',
    '\n',
    '# ── ETAPA 6.2 — Predicciones con modelo ganador (LightGBM Etapa 4) ──────────\n',
    "modelo_final = resultados_opt['LightGBM']['modelo']\n",
    "umbral_final = resultados_opt['LightGBM']['Umbral']\n",
    '\n',
    'y_test_proba = modelo_final.predict_proba(X_test_final)[:, 1]\n',
    '\n',
    'print(f"Predicciones generadas: {len(y_test_proba):,} transacciones")\n',
    'print(f"Distribución de probabilidades predichas:")\n',
    'print(f"  Media:     {y_test_proba.mean():.4f}")\n',
    'print(f"  Mediana:   {np.median(y_test_proba):.4f}")\n',
    'print(f"  Min / Max: {y_test_proba.min():.4f} / {y_test_proba.max():.4f}")\n',
    'print()\n',
    '\n',
    '# Etiquetas binarias con umbral óptimo (para referencia interna, no Kaggle)\n',
    'y_test_binary = (y_test_proba >= umbral_final).astype(int)\n',
    'tasa_fraude_pred = y_test_binary.mean() * 100\n',
    'print(f"Con umbral óptimo ({umbral_final:.2f}): {y_test_binary.sum():,} transacciones '
    'clasificadas como fraude ({tasa_fraude_pred:.2f}%)")\n',
    'print()\n',
    '\n',
    '# ── ETAPA 6.3 — Creación del archivo de submission ──────────────────────────\n',
    '# Kaggle evalúa con AUC-ROC → submission contiene probabilidades, no etiquetas\n',
    'submission = pd.DataFrame({\n',
    "    'TransactionID': df_test['TransactionID'].values,\n",
    "    'isFraud'      : y_test_proba\n",
    '})\n',
    '\n',
    "submission.to_csv('submission.csv', index=False)\n",
    'print(f"submission.csv guardado: {submission.shape[0]:,} filas")\n',
    'print()\n',
    'display(submission.head(10))\n',
])

# ── New Cell A: Figura 8 markdown ─────────────────────────────────────────────
cell_fig8_md = mk_md([
    '### Figura 8. Distribución de probabilidades predichas — Test Kaggle vs. Validación\n',
    '\n',
    'Comparativa de las distribuciones de probabilidad de fraude predichas por el modelo '
    'LightGBM ganador sobre el conjunto de test Kaggle y el conjunto de validación. '
    'La alineación entre ambas distribuciones valida la consistencia del modelo en datos no vistos.\n',
])

# ── New Cell B: Figura 8 code ─────────────────────────────────────────────────
cell_fig8 = mk_code([
    '# ── Figura 8 — Distribución de probabilidades: Test vs. Validación ──────────\n',
    'import matplotlib.pyplot as plt\n',
    'import numpy as np\n',
    '\n',
    '# Probabilidades en validación (del modelo LightGBM Etapa 4)\n',
    "y_val_proba = resultados_opt['LightGBM']['modelo'].predict_proba(X_val)[:, 1]\n",
    '\n',
    'fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n',
    '\n',
    '# ── Panel izquierdo: histograma completo ─────────────────────────────────────\n',
    "axes[0].hist(y_val_proba,   bins=60, alpha=0.6, color='#1565C0', label='Validación')\n",
    "axes[0].hist(y_test_proba,  bins=60, alpha=0.6, color='#2E7D32', label='Test Kaggle')\n",
    "axes[0].axvline(umbral_final, color='red', linestyle='--', linewidth=1.5,\n",
    "                label=f'Umbral óptimo ({umbral_final:.2f})')\n",
    "axes[0].set_xlabel('Probabilidad predicha de fraude')\n",
    "axes[0].set_ylabel('Frecuencia')\n",
    "axes[0].set_title('Distribución completa')\n",
    "axes[0].legend()\n",
    "axes[0].set_yscale('log')\n",
    '\n',
    '# ── Panel derecho: zona de alta probabilidad (zoom >0.3) ─────────────────────\n',
    "mask_val  = y_val_proba  >= 0.3\n",
    "mask_test = y_test_proba >= 0.3\n",
    "axes[1].hist(y_val_proba[mask_val],   bins=40, alpha=0.6, color='#1565C0', label='Validación')\n",
    "axes[1].hist(y_test_proba[mask_test], bins=40, alpha=0.6, color='#2E7D32', label='Test Kaggle')\n",
    "axes[1].axvline(umbral_final, color='red', linestyle='--', linewidth=1.5,\n",
    "                label=f'Umbral óptimo ({umbral_final:.2f})')\n",
    "axes[1].set_xlabel('Probabilidad predicha de fraude')\n",
    "axes[1].set_ylabel('Frecuencia')\n",
    "axes[1].set_title('Zoom: probabilidad ≥ 0.30')\n",
    "axes[1].legend()\n",
    '\n',
    'plt.suptitle(\n',
    "    'Figura 8. Distribución de probabilidades predichas — LightGBM (Etapa 4)',\n",
    "    fontsize=11, fontweight='bold', y=1.01\n",
    ')\n',
    'plt.tight_layout()\n',
    "fig_path = 'fig_08_distribucion_probabilidades.png'\n",
    'plt.savefig(fig_path, dpi=150, bbox_inches=\'tight\')\n',
    'plt.show()\n',
    'print(f"Figura 8 guardada en {fig_path}")\n',
])

# ── New Cell C: Interpretation placeholder ────────────────────────────────────
cell_interp6 = mk_md([
    '#### Resultados — Generación de Predicciones Finales\n',
    '\n',
    '*[Esta sección se completará con los resultados reales tras ejecutar las celdas anteriores. '
    'Comparte el output de la celda de predicciones y la Figura 8.]*\n',
])

# ── Insert new cells after [68] ───────────────────────────────────────────────
new_cells = [cell_fig8_md, cell_fig8, cell_interp6]
nb['cells'] = nb['cells'][:69] + new_cells

with open('Codigo-TFM_Ariel_Mery_Ibarra.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Syntax check on new code cells
print(f'Total celdas: {len(nb["cells"])}')
for idx in [68, 70]:
    src = ''.join(nb['cells'][idx]['source'])
    try:
        compile(src, f'cell_{idx}', 'exec')
        print(f'Cell [{idx}] OK')
    except SyntaxError as e:
        print(f'Cell [{idx}] ERROR line {e.lineno}: {e}')
        lines = src.split('\n')
        for j in range(max(0,e.lineno-2), min(len(lines),e.lineno+2)):
            print(f'  {j+1}: {repr(lines[j])}')
