        c1.metric("MSE (test)", f"{mse_rf:.2f}")
        c2.metric("RMSE (test)", f"{rmse_rf:.2f}")
        c3.metric("MAE (test)", f"{mae_rf:.2f}")
        c4.metric("R² (test)", f"{r2_rf:.3f}")

        # Importancia
        imp = pd.Series(rf.feature_importances_, index=columnas_entrada).sort_values(ascending=False)
        fig_imp, ax = plt.subplots(figsize=(6,4))
        imp.plot(kind="bar", ax=ax)
        ax.set_title("Importancia de características (RF)")
        ax.set_ylabel("Importancia")
        ax.grid(True, axis="y", alpha=0.4)
        fig_imp.tight_layout()
        st.pyplot(fig_imp, use_container_width=True)

        st.dataframe(imp.reset_index().rename(columns={"index":"Feature", 0:"Importance"}), use_container_width=True)
    else:
        st.info("Activa **RandomForestRegressor** para visualizar.")

# ===========================
# 4) K-FOLD CV (NN REGRESIÓN)
# ===========================
with tabs[3]:
    st.subheader("🔁 Validación Cruzada (K=5) para NN de Regresión")

    if RUN_KFOLD:
        X_raw = X.copy()
        y_raw = y_real.copy()

        def nn_builder(input_dim):
            m = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.15),
                layers.Dense(64, activation="relu"),
                layers.Dense(1)
            ])
            m.compile(optimizer=keras.optimizers.Adam(), loss=Huber(), metrics=["mae"])
            return m

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics = []

        progress = st.progress(0.0, text="Ejecutando K-Fold...")
        for fold, (tr_idx, te_idx) in enumerate(kf.split(X_raw), start=1):
            X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
            y_tr, y_te = y_raw[tr_idx], y_raw[te_idx]

            y_tr_t = np.log1p(y_tr) if Y_TRANSFORM == "log1p" else y_tr.copy()
            y_te_t = np.log1p(y_te) if Y_TRANSFORM == "log1p" else y_te.copy()

            scaler_cv = RobustScaler() if USE_ROBUST_SCALER else StandardScaler()
            X_tr_s = scaler_cv.fit_transform(X_tr)
            X_te_s = scaler_cv.transform(X_te)

            model_cv = nn_builder(X_tr_s.shape[1])
            es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
            rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0)
            model_cv.fit(X_tr_s, y_tr_t, validation_split=0.2, epochs=300, batch_size=32, verbose=0, callbacks=[es, rl])

            y_pred_t = model_cv.predict(X_te_s, verbose=0).ravel()
            y_pred = np.expm1(y_pred_t) if Y_TRANSFORM == "log1p" else y_pred_t

            mse  = mean_squared_error(y_te, y_pred)
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_te, y_pred)
            r2   = r2_score(y_te, y_pred)
            metrics.append((fold, mse, rmse, mae, r2))

            progress.progress(fold/5.0, text=f"Fold {fold}/5 completado")

        df_cv = pd.DataFrame(metrics, columns=["Fold","MSE","RMSE","MAE","R2"])
        st.dataframe(df_cv.style.format({"MSE":"{:.2f}","RMSE":"{:.2f}","MAE":"{:.2f}","R2":"{:.3f}"}),
                     use_container_width=True)

        st.write("**Promedios ± std:**")
        st.write(f"- MSE : {df_cv['MSE'].mean():.2f} ± {df_cv['MSE'].std():.2f}")
        st.write(f"- RMSE: {df_cv['RMSE'].mean():.2f} ± {df_cv['RMSE'].std():.2f}")
        st.write(f"- MAE : {df_cv['MAE'].mean():.2f} ± {df_cv['MAE'].std():.2f}")
        st.write(f"- R²  : {df_cv['R2'].mean():.3f} ± {df_cv['R2'].std():.3f}")

        st.download_button("⬇️ Descargar métricas K-Fold (CSV)",
                           data=df_cv.to_csv(index=False).encode("utf-8"),
                           file_name="kfold_metrics.csv", mime="text/csv")
    else:
        st.info("Activa **K-Fold CV (NN)** para visualizar.")

# ===========================
# 5) CLASIFICACIÓN DIRECTA (SVM/KNN, 4 clases)
# ===========================
with tabs[4]:
    st.subheader("🎯 Clasificación directa (SVM/KNN) — 4 clases")

    if RUN_CLF:
        bins = [0, 2, 7, 40, np.inf]
        labels_bins = ["Muy bajo (0–2)", "Bajo (2–7)", "Moderado (7–40)", "Muy alto (≥40)"]

        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X, y_real, test_size=0.20, random_state=42
        )
        y_train_cls = pd.cut(y_train_rf, bins=bins, labels=labels_bins, right=False)
        y_test_cls  = pd.cut(y_test_rf,  bins=bins, labels=labels_bins, right=False)

        svm_clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced",
                probability=True, random_state=42)
        )
        knn_clf = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights="distance")
        )

        with st.spinner("Entrenando SVM y KNN..."):
            svm_clf.fit(X_train_rf, y_train_cls)
            knn_clf.fit(X_train_rf, y_train_cls)

        y_pred_svm = svm_clf.predict(X_test_rf)
        y_pred_knn = knn_clf.predict(X_test_rf)

        cm_svm = confusion_matrix(y_test_cls, y_pred_svm, labels=labels_bins)
        cm_knn = confusion_matrix(y_test_cls, y_pred_knn, labels=labels_bins)

        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(plot_confusion_matrix_pretty(cm_svm, labels_bins, "Matriz de confusión — SVM (4 clases)"),
                      use_container_width=True)
            st.code(classification_report(
                y_test_cls, y_pred_svm, labels=labels_bins, target_names=labels_bins,
                digits=3, zero_division=0
            ))
        with col_b:
            st.pyplot(plot_confusion_matrix_pretty(cm_knn, labels_bins, "Matriz de confusión — KNN (4 clases)"),
                      use_container_width=True)
            st.code(classification_report(
                y_test_cls, y_pred_knn, labels=labels_bins, target_names=labels_bins,
                digits=3, zero_division=0
            ))

        # --------- MATRICES DIFUSAS CON PROBABILIDADES ---------
        if USE_FUZZY:
            proba_svm = svm_clf.predict_proba(X_test_rf)
            proba_knn = knn_clf.predict_proba(X_test_rf)

            _, svm_classes = get_classes_from_pipeline(svm_clf)
            _, knn_classes = get_classes_from_pipeline(knn_clf)

            proba_svm_al = align_proba_to_labels(proba_svm, svm_classes, labels_bins)
            proba_knn_al = align_proba_to_labels(proba_knn, knn_classes, labels_bins)

            cm_svm_fuzzy = fuzzy_confusion_from_probs(y_test_rf, proba_svm_al, n_classes=4, eps=EPS)
            cm_knn_fuzzy = fuzzy_confusion_from_probs(y_test_rf, proba_knn_al, n_classes=4, eps=EPS)

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.pyplot(
                    plot_confusion_matrix_pretty_float(cm_svm_fuzzy, labels_bins,
                                                       "Matriz **difusa** — SVM (con probas)"),
                    use_container_width=True
                )
                st.caption(f"Suma de pesos (SVM): {cm_svm_fuzzy.sum():.2f}")
            with col_f2:
                st.pyplot(
                    plot_confusion_matrix_pretty_float(cm_knn_fuzzy, labels_bins,
                                                       "Matriz **difusa** — KNN (con probas)"),
                    use_container_width=True
                )
                st.caption(f"Suma de pesos (KNN): {cm_knn_fuzzy.sum():.2f}")
        # --------------------------------------------------------
    else:
        st.info("Activa **Clasificación directa (SVM/KNN)** para visualizar.")

# ===========================
# 6) 🧐 VISUALIZACIÓN DE NUEVAS PREDICCIONES
# ===========================
with tabs[5]:
    st.subheader("🧐 Visualización de nuevas predicciones")
    st.caption("Sube un CSV con **solo**: pH, Temperatura (°C), Conductividad (μS/cm), Oxígeno Disuelto (mg/L), Turbidez (NTU).")

    if not TRY_NEW_DATA:
        st.info("Activa **‘Probar modelo con datos nuevos’** en el panel lateral para habilitar esta pestaña.")
        st.stop()

    def read_csv_robust(uploaded):
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        seps = [None, ",", ";", "\t", "|"]
        for enc in encodings:
            for sep in seps:
                try:
                    uploaded.seek(0)
                    df = pd.read_csv(
                        uploaded, encoding=enc, sep=sep,
                        engine="python" if sep is None else "c",
                    )
                    return df
                except Exception:
                    continue
        return None

    model_infer = model if "model" in locals() else None
    scaler_infer = scaler if "scaler" in locals() else None

    if model_infer is None or scaler_infer is None:
        st.warning("No encuentro un modelo/scaler entrenados en esta sesión. Sube tus archivos guardados.")
        mdl_file = st.file_uploader("Modelo (.keras/.h5)", type=["keras", "h5", "hdf5"], key="mdl_new")
        scl_file = st.file_uploader("Scaler (.pkl)", type=["pkl"], key="scl_new")
        if mdl_file and scl_file:
            try:
                model_infer = keras.models.load_model(mdl_file)
                scaler_infer = joblib.load(scl_file)
                st.success("Modelo y scaler cargados correctamente.")
            except Exception as e:
                st.error(f"No se pudo cargar el modelo/scaler: {e}")
                st.stop()
        else:
            st.info("Sube **ambos** archivos para continuar.")
            st.stop()

    up_csv = st.file_uploader("Cargar CSV con **nuevos** datos (sin clorofila)", type=["csv"], key="csv_newdata")
    if up_csv is None:
        st.info("Sube un CSV para ver matrices y tabla de resultados.")
        st.stop()

    df_new = read_csv_robust(up_csv)
    if df_new is None or df_new.empty:
        st.error("No pude leer el CSV. Guarda como UTF-8 (o UTF-8 con BOM) y usa coma/; como separador.")
        st.stop()

    st.write("Vista previa del archivo cargado:")
    st.dataframe(df_new.head(), use_container_width=True)

    def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        def canon(s: str) -> str:
            s = str(s)
            s = unicodedata.normalize("NFKD", s)
            s = s.replace("µ", "u").replace("μ", "u").replace("?", "u")
            s = "".join(ch for ch in s if ch.isprintable())
            s = s.lower().strip()
            s = re.sub(r"\s+", " ", s)
            return s

        direct = {
            "ph": "pH",
            "temperatura": "Temperatura (°C)",
            "temperatura (c)": "Temperatura (°C)",
            "temp (°c)": "Temperatura (°C)",
            "temp": "Temperatura (°C)",

            "conductividad (us/cm)": "Conductividad (μS/cm)",
            "conductividad(us/cm)": "Conductividad (μS/cm)",
            "conductividad (s/cm)":  "Conductividad (μS/cm)",
            "conductividad (?s/cm)": "Conductividad (μS/cm)",
            "conductividad": "Conductividad (μS/cm)",

            "oxigeno disuelto (mg/l)": "Oxígeno Disuelto (mg/L)",
            "oxígeno disuelto (mg/l)": "Oxígeno Disuelto (mg/L)",
            "oxigeno disuelto": "Oxígeno Disuelto (mg/L)",
            "do (mg/l)": "Oxígeno Disuelto (mg/L)",

            "turbidez (ntu)": "Turbidez (NTU)",
            "turbiedad (ntu)": "Turbidez (NTU)",
            "turbidez": "Turbidez (NTU)",
            "turbiedad": "Turbidez (NTU)",
        }

        renamed = {}
        for c in df.columns:
