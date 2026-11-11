# ESTADÍSTICA APLICADA - CASO ESTUDIO 1 - GRUPO 11 
# Predicción de la demanda de electricidad de España (MW) con 24 h de anticipación (desde 2015 a 2018)

# 1.- INSTALACION LIBRERIAS 

rm(list = ls())

# Cargar los paquetes. Si alguno no está instalado, instalarlo una sola vez con
# install.packages("nombre"), por ejemplo: install.packages("gbm")
library(tidyverse)    # leer datos, transformar tablas y hacer gráficos
library(lubridate)    # trabajar con fechas y horas
library(glmnet)       # ridge, lasso y elastic-net
library(mgcv)         # GAM (curvas suaves)
library(rpart)        # árbol de regresión
library(rpart.plot)   # dibujo del árbol
library(ranger)       # Random Forest
library(gbm)          # boosting
library(pdp)          # gráficos de dependencia parcial (PDP/ICE)
library(lmtest)       # pruebas formales: Breusch-Pagan, Durbin-Watson

# Semilla: hace que todo lo que use azar dé siempre el mismo resultado
set.seed(2026)

# Estilo de todos los gráficos
theme_set(theme_minimal(base_size = 12))

# La carpeta de trabajo debe ser "equipo_regresion"
# (Session -> Set Working Directory -> Choose Directory)
if (!file.exists("datos/energy_dataset.csv")) {
  stop("No se encontró datos/energy_dataset.csv. Fije la carpeta de trabajo en ",
       "'equipo_regresion' con Session -> Set Working Directory.")
}
dir.create("resultados", showWarnings = FALSE)   # aquí se guardan tablas y gráficos


# 1. LECTURA Y AUDITORÍA DE DATOS
energia_cruda <- read_csv("datos/energy_dataset.csv",
                          col_types = cols(time = col_character(), .default = col_double()))

columnas_necesarias <- c("time", "total load actual", "total load forecast",
                         "price day ahead", "generation solar",
                         "generation wind onshore", "generation fossil gas")
columnas_faltantes <- setdiff(columnas_necesarias, names(energia_cruda))
if (length(columnas_faltantes) > 0) {
  stop("Faltan columnas en el archivo: ", paste(columnas_faltantes, collapse = ", "))
}

# Auditoría: tamaño, datos faltantes o duplicados y regularidad horaria
auditoria <- tibble(
  filas            = nrow(energia_cruda),
  columnas         = ncol(energia_cruda),
  desde            = min(energia_cruda$time),
  hasta            = max(energia_cruda$time),
  horas_duplicadas = sum(duplicated(energia_cruda$time)),
  na_demanda       = sum(is.na(energia_cruda$`total load actual`)),
  columnas_vacias  = sum(colSums(is.na(energia_cruda)) == nrow(energia_cruda))
)
print(auditoria)
write_csv(auditoria, "resultados/tabla_auditoria.csv")

# Faltantes por variable
faltantes_por_variable <- energia_cruda %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_faltantes") %>%
  arrange(desc(n_faltantes))
write_csv(faltantes_por_variable, "resultados/tabla_faltantes.csv")


# 2. CONSTRUCCIÓN DE VARIABLES

# Se arman los predictores para predecir la demanda con 24 h de anticipación:
# calendario (hora, día, mes, festivos) y valores de hace 24 h o de hace una semana.
# La generación de la misma hora no se usa: es casi igual a la demanda (leakage).


# Festivos nacionales de España 2015-2018 (afectan fuertemente el consumo)
festivos_nacionales <- as.Date(c(
  "2015-01-01", "2015-01-06", "2015-04-03", "2015-05-01", "2015-08-15", "2015-10-12",
  "2015-11-01", "2015-12-06", "2015-12-08", "2015-12-25",
  "2016-01-01", "2016-01-06", "2016-03-25", "2016-05-01", "2016-08-15", "2016-10-12",
  "2016-11-01", "2016-12-06", "2016-12-08", "2016-12-25",
  "2017-01-01", "2017-01-06", "2017-04-14", "2017-05-01", "2017-08-15", "2017-10-12",
  "2017-11-01", "2017-12-06", "2017-12-08", "2017-12-25",
  "2018-01-01", "2018-01-06", "2018-03-30", "2018-05-01", "2018-08-15", "2018-10-12",
  "2018-11-01", "2018-12-06", "2018-12-08", "2018-12-25"))

datos_completos <- energia_cruda %>%
  transmute(
    # Si 'time' llegara ya como fecha-hora (lectura sin col_types), ymd_hms() fallaría justo en
    # las medianoches; por eso se usa directamente. Si es texto, se parsea (el offset +01:00 se
    # convierte a UTC).
    fecha_utc      = if (inherits(time, "POSIXct")) with_tz(time, "UTC") else ymd_hms(time),
    demanda        = `total load actual`,           # RESPUESTA (MW)
    pronostico_tso = `total load forecast`,         # pronóstico del operador 
    precio_da      = `price day ahead`,
    gen_solar      = `generation solar`,
    gen_eolica     = `generation wind onshore`,
    gen_gas        = `generation fossil gas`
  ) %>%
  arrange(fecha_utc) %>%
  mutate(
    fecha_local  = with_tz(fecha_utc, "Europe/Madrid"),
    fecha        = as_date(fecha_local),
    hora         = hour(fecha_local),
    dia_semana   = wday(fecha_local, week_start = 1),   # 1 = lunes ... 7 = domingo
    mes          = month(fecha_local),
    dia_anio     = yday(fecha_local),
    no_laborable = as.integer(dia_semana >= 6 | fecha %in% festivos_nacionales),
    # Versiones como factor (niveles fijos) para los modelos lineales
    hora_f       = factor(hora, levels = 0:23),
    dia_f        = factor(dia_semana, levels = 1:7,
                          labels = c("lun", "mar", "mie", "jue", "vie", "sab", "dom")),
    mes_f        = factor(mes, levels = 1:12),
    tipo_dia     = factor(ifelse(no_laborable == 1, "no_laborable", "laborable"),
                          levels = c("laborable", "no_laborable")),
    # Rezagos: información del pasado (>= 24 h), conocida al momento de predecir
    demanda_lag24  = lag(demanda, 24),     # misma hora, día anterior
    demanda_lag168 = lag(demanda, 168),    # misma hora, semana anterior
    demanda_prom24 = lag(as.numeric(stats::filter(demanda, rep(1 / 24, 24), sides = 1)), 24),
    precio_lag24   = lag(precio_da, 24),
    solar_lag24    = lag(gen_solar, 24),
    eolica_lag24   = lag(gen_eolica, 24),
    gas_lag24      = lag(gen_gas, 24)
  )

# Chequeo temprano: sin NA en la hora y serie estrictamente horaria. Si falla, los rezagos
# (filas anteriores) quedarían desalineados y TODOS los resultados serían inválidos.
stopifnot(
  "Hay horas (fecha_utc) sin parsear: revise la lectura de la columna 'time'" =
    !anyNA(datos_completos$fecha_utc),
  "La serie no es estrictamente horaria: los rezagos quedarían desalineados" =
    all(diff(as.numeric(datos_completos$fecha_utc)) == 3600))

# Unidad de análisis: una hora del sistema eléctrico español.
# Se descarta la primera semana (sin rezago de 168 h (valor de la misma variable exactamente una semana antes,)) y las horas sin demanda observada.
datos <- datos_completos %>%
  filter(row_number() > 168, !is.na(demanda))

predictores <- c("hora", "dia_semana", "dia_anio", "no_laborable",
                 "demanda_lag24", "demanda_lag168", "demanda_prom24",
                 "precio_lag24", "solar_lag24", "eolica_lag24", "gas_lag24")

cat("Observaciones utilizables:", nrow(datos), "| Predictores:", length(predictores), "\n")



# 3.- PARTICIÓN TEMPORAL

# El modelo se usará para predecir el FUTURO, así que la partición es temporal:
# se entrena con el pasado (2015-2017) y se prueba con 2018. (test se usa solo al final)

fecha_corte <- as.Date("2017-12-31")   # último día de entrenamiento
entrenamiento <- datos %>% filter(fecha <= fecha_corte)

# Folds de validación cruzada temporal: cada fold valida un trimestre de 2017 y
# entrena con todo lo anterior, dejando 1 día de separación ("purga") porque
# el pronóstico se hace con 24 h de anticipación.
inicios_folds <- as.Date(c("2017-01-01", "2017-04-01", "2017-07-01", "2017-10-01"))
folds <- map(1:4, function(k) {
  inicio <- inicios_folds[k]
  fin <- if (k < 4) inicios_folds[k + 1] - 1 else fecha_corte
  list(entrenar = which(entrenamiento$fecha < inicio - 1),
       validar  = which(entrenamiento$fecha >= inicio & entrenamiento$fecha <= fin))
})
tabla_folds <- map_dfr(seq_along(folds), function(k) tibble(
  fold             = k,
  entrena_desde    = min(entrenamiento$fecha[folds[[k]]$entrenar]),
  entrena_hasta    = max(entrenamiento$fecha[folds[[k]]$entrenar]),
  valida_desde     = min(entrenamiento$fecha[folds[[k]]$validar]),
  valida_hasta     = max(entrenamiento$fecha[folds[[k]]$validar]),
  n_entrenamiento  = length(folds[[k]]$entrenar),
  n_validacion     = length(folds[[k]]$validar)))
print(tabla_folds)
write_csv(tabla_folds, "resultados/tabla_folds.csv")

# Gráfico de la serie con la partición. Sólo se dibuja el ENTRENAMIENTO: el período
# de test se marca con una banda gris, sin mostrar su demanda (el test no se mira
# hasta la sección 9).
fin_test <- as.Date("2018-12-31")
g_serie <- entrenamiento %>%
  group_by(fecha) %>% summarise(demanda = mean(demanda), .groups = "drop") %>%
  ggplot(aes(fecha, demanda)) +
  annotate("rect", xmin = fecha_corte + 1, xmax = fin_test,
           ymin = -Inf, ymax = Inf, fill = "grey80", alpha = 0.6) +
  annotate("text", x = fecha_corte + 1 + as.numeric(fin_test - fecha_corte) / 2,
           y = Inf, vjust = 1.6, label = "Test (2018)\nsin usar hasta el final") +
  geom_line(linewidth = 0.3, colour = "#0072B2") +
  geom_vline(xintercept = inicios_folds, linetype = "dashed", colour = "grey40") +
  labs(title = "Demanda eléctrica diaria promedio en España (entrenamiento 2015-2017)",
       subtitle = "Líneas punteadas: inicio de cada fold de validación (trimestres de 2017)",
       x = NULL, y = "Demanda promedio (MW)")
ggsave("resultados/fig01_serie_y_particion.png", g_serie, width = 9, height = 4.5, dpi = 150)

g_perfil <- entrenamiento %>%
  group_by(hora, tipo_dia) %>% summarise(demanda = mean(demanda), .groups = "drop") %>%
  ggplot(aes(hora, demanda, colour = tipo_dia)) + geom_line(linewidth = 1) +
  labs(title = "Perfil diario de la demanda (entrenamiento)",
       x = "Hora del día", y = "Demanda promedio (MW)", colour = NULL) +
  theme(legend.position = "bottom")
ggsave("resultados/fig02_perfil_horario.png", g_perfil, width = 7, height = 4, dpi = 150)



# 4. FUNCIONES DE APOYO

# Métricas de error: RMSE y MAE en MW; R² = proporción de varianza explicada
metricas <- function(y, pred) {
  tibble(RMSE = sqrt(mean((y - pred)^2)),
         MAE  = mean(abs(y - pred)),
         R2   = 1 - sum((y - pred)^2) / sum((y - mean(y))^2))
}

# Imputación por mediana DENTRO de cada fold: las medianas se calculan sólo con
# los datos de ajuste y se aplican a los datos nuevos (evita preprocesamiento global).
imputar_mediana <- function(ajuste, nuevos, variables = predictores) {
  medianas <- map_dbl(ajuste[variables], median, na.rm = TRUE)
  for (v in variables) {
    ajuste[[v]][is.na(ajuste[[v]])] <- medianas[[v]]
    nuevos[[v]][is.na(nuevos[[v]])] <- medianas[[v]]
  }
  list(ajuste = ajuste, nuevos = nuevos)
}

# Validación cruzada temporal genérica.
# `ajustar_y_predecir(ajuste, nuevos)` entrena un modelo y devuelve una matriz de
# predicciones: una columna por cada combinación de hiperparámetros.
validacion_temporal <- function(nombre, ajustar_y_predecir) {
  inicio_reloj <- Sys.time()
  resultado <- map_dfr(seq_along(folds), function(k) {
    imp <- imputar_mediana(entrenamiento[folds[[k]]$entrenar, ],
                           entrenamiento[folds[[k]]$validar, ])
    pred <- ajustar_y_predecir(imp$ajuste, imp$nuevos)
    as_tibble(pred) %>%
      mutate(id = folds[[k]]$validar, fold = k,
             y = imp$nuevos$demanda, no_laborable = imp$nuevos$no_laborable) %>%
      pivot_longer(-c(id, fold, y, no_laborable), names_to = "config", values_to = "pred")
  }) %>% mutate(modelo = nombre)
  cat(sprintf("  %-22s listo en %5.1f s\n", nombre,
              as.numeric(difftime(Sys.time(), inicio_reloj, units = "secs"))))
  resultado
}

# Resume la CV: métricas por fold y luego promedio, desviación y error estándar
resumir_cv <- function(pred_cv) {
  pred_cv %>%
    group_by(modelo, config, fold) %>%
    summarise(RMSE     = sqrt(mean((y - pred)^2)),
              MAE      = mean(abs(y - pred)),
              R2       = 1 - sum((y - pred)^2) / sum((y - mean(y))^2),
              RMSE_sub = sqrt(mean(((y - pred)[no_laborable == 1])^2)),
              .groups = "drop") %>%
    group_by(modelo, config) %>%
    summarise(DE_RMSE  = sd(RMSE),
              EE_RMSE  = sd(RMSE) / sqrt(n()),
              RMSE     = mean(RMSE),
              MAE      = mean(MAE),
              R2       = mean(R2),
              RMSE_sub = mean(RMSE_sub),
              .groups = "drop")
}

# Fórmulas de los modelos
#  - Lineales: hora, día y mes como factores; interacción hora x tipo de día
formula_lineal <- demanda ~ hora_f * no_laborable + dia_f + mes_f +
  demanda_lag24 + demanda_lag168 + demanda_prom24 +
  precio_lag24 + solar_lag24 + eolica_lag24 + gas_lag24
#  - Árboles: usan las variables numéricas directamente (encuentran cortes solos)
formula_arboles <- reformulate(predictores, response = "demanda")

matriz_x <- function(df) model.matrix(formula_lineal, data = df)[, -1]

# Funciones "ajustar y predecir" de cada modelo
ap_baseline <- function(ajuste, nuevos) {
  # Baseline: promedio histórico de la demanda por hora y día de la semana
  perfil <- ajuste %>% group_by(hora, dia_semana) %>%
    summarise(pred = mean(demanda), .groups = "drop")
  cbind(promedio_hora_dia = left_join(nuevos, perfil, by = c("hora", "dia_semana"))$pred)
}

ap_ols <- function(ajuste, nuevos) {
  cbind(ols = predict(lm(formula_lineal, data = ajuste), newdata = nuevos))
}

# Grilla de 60 valores de lambda (fuerza del castigo), de 10.000 a 0,01 en escala logarítmica
grilla_lambda <- 10^seq(4, -2, length.out = 60)

# alpha: 0 = ridge, 0.5 = elastic-net, 1 = lasso
ap_glmnet <- function(ajuste, nuevos, alphas = c(0, 0.5, 1)) {
  # glmnet estandariza las variables internamente con los datos de AJUSTE del fold
  x_ajuste <- matriz_x(ajuste)
  x_nuevos <- matriz_x(nuevos)
  map(alphas, function(a) {
    ajuste_glmnet <- glmnet(x_ajuste, ajuste$demanda, alpha = a, lambda = grilla_lambda)
    p <- predict(ajuste_glmnet, newx = x_nuevos)
    colnames(p) <- sprintf("alpha=%.1f|lambda=%.4g", a, grilla_lambda)
    p
  }) %>% do.call(cbind, .)
}

formula_gam <- function(k) {
  as.formula(sprintf(paste(
    "demanda ~ tipo_dia + dia_f + s(hora, by = tipo_dia, bs = 'cc', k = 20)",
    "+ s(dia_anio, bs = 'cc', k = %d) + s(demanda_lag24, k = %d) + s(demanda_lag168, k = %d)",
    "+ s(demanda_prom24, k = %d) + s(precio_lag24, k = %d) + s(solar_lag24, k = %d)",
    "+ s(eolica_lag24, k = %d) + s(gas_lag24, k = %d)"), k, k, k, k, k, k, k, k))
}
ajustar_gam <- function(ajuste, k) {
  # bam = gam para muchos datos; splines cíclicos ('cc') para hora y día del año
  bam(formula_gam(k), data = ajuste, method = "fREML", discrete = TRUE,
      knots = list(hora = c(-0.5, 23.5), dia_anio = c(0.5, 366.5)))
}
# k = nº de funciones base (nudos) de cada curva; se prueban 5 y 10
ap_gam <- function(ajuste, nuevos, ks = c(5, 10)) {
  map(ks, function(k) {
    matrix(predict(ajustar_gam(ajuste, k), newdata = nuevos),
           dimnames = list(NULL, paste0("k=", k)))
  }) %>% do.call(cbind, .)
}

# cp = complejidad del árbol: más pequeño = árbol más grande
ap_arbol <- function(ajuste, nuevos, cps = c(0.01, 0.001, 0.0001)) {
  map(cps, function(cp) {
    arbol <- rpart(formula_arboles, data = ajuste, method = "anova",
                   control = rpart.control(cp = cp, xval = 0, minbucket = 20))
    matrix(predict(arbol, nuevos), dimnames = list(NULL, paste0("cp=", cp)))
  }) %>% do.call(cbind, .)
}

# mtry = nº de variables al azar que mira cada corte; 300 árboles
ap_rf <- function(ajuste, nuevos, mtrys = c(3, 6, 9)) {
  map(mtrys, function(m) {
    bosque <- ranger(formula_arboles, data = ajuste, num.trees = 300,
                     mtry = m, min.node.size = 5, seed = 2026)
    matrix(predict(bosque, nuevos)$predictions, dimnames = list(NULL, paste0("mtry=", m)))
  }) %>% do.call(cbind, .)
}

# Boosting: profundidad 3 o 5 y shrinkage (tasa de aprendizaje) 0.05, 0.1 o 0.2, ambos
# validados por CV; hasta 1.000 árboles (se evalúa cada 100). Con shrinkage más
# alto se necesitan menos árboles, por eso el número de árboles también se elige por CV.
ap_gbm <- function(ajuste, nuevos, profundidades = c(3, 5), shrinkages = c(0.05, 0.1, 0.2)) {
  n_arboles <- seq(100, 1000, by = 100)
  grilla_gbm <- expand.grid(d = profundidades, s = shrinkages)
  map(seq_len(nrow(grilla_gbm)), function(i) {
    d <- grilla_gbm$d[i]
    sh <- grilla_gbm$s[i]
    set.seed(2026)
    boost <- gbm(formula_arboles, data = as.data.frame(ajuste[, c("demanda", predictores)]),
                 distribution = "gaussian", n.trees = 1000,
                 interaction.depth = d, shrinkage = sh,
                 n.minobsinnode = 20, bag.fraction = 0.5, verbose = FALSE)
    p <- predict(boost, as.data.frame(nuevos), n.trees = n_arboles)
    colnames(p) <- sprintf("prof=%d|shr=%.2f|arboles=%d", d, sh, n_arboles)
    p
  }) %>% do.call(cbind, .)
}

# Catálogo de modelos: nombre, familia, complejidad (para la regla de parsimonia)
catalogo <- tribble(
  ~modelo,              ~familia,       ~complejidad, ~funcion,
  "Baseline",           "—",            1,            "ap_baseline",
  "OLS",                "Lineal",       2,            "ap_ols",
  "Ridge/Lasso/EN",     "Regularizado", 3,            "ap_glmnet",
  "GAM",                "Flexible",     4,            "ap_gam",
  "Árbol",              "Árbol",        5,            "ap_arbol",
  "Random Forest",      "Ensemble",     6,            "ap_rf",
  "Boosting",           "Ensemble",     7,            "ap_gbm",
  "Averaging",          "Ensemble",     8,            NA_character_
)


# 5. OLS: INFERENCIA, SIGNIFICANCIA vs RELEVANCIA Y DIAGNÓSTICO
# Se ajusta el OLS con TODO el entrenamiento (2015-2017; el test no participa).

imp_total <- imputar_mediana(entrenamiento, entrenamiento)
entrenamiento_imp <- imp_total$ajuste
modelo_ols <- lm(formula_lineal, data = entrenamiento_imp)
cat(sprintf("\nOLS: R2 = %.3f | R2 ajustado = %.3f | n = %d\n",
            summary(modelo_ols)$r.squared, summary(modelo_ols)$adj.r.squared, nobs(modelo_ols)))

#     Intervalos de confianza de los coeficientes de las variables numéricas.
#     Relevancia = efecto (en MW) de aumentar la variable en 1 desviación estándar,
#     comparado con la demanda media. Un p-valor pequeño NO implica un efecto grande.
numericas <- c("demanda_lag24", "demanda_lag168", "demanda_prom24",
               "precio_lag24", "solar_lag24", "eolica_lag24", "gas_lag24")
ic_ols <- confint(modelo_ols, level = 0.95)

# IC por bootstrap de bloques diarios: los residuos están autocorrelacionados
# (horas consecutivas se parecen), lo que hace que el IC clásico sea demasiado angosto.
set.seed(2026)
dias_entrenamiento <- unique(entrenamiento_imp$fecha)
filas_por_dia <- split(seq_len(nrow(entrenamiento_imp)), entrenamiento_imp$fecha)
boot_coef <- map_dfr(seq_len(200), function(b) {   # 200 remuestreos
  dias_b <- sample(as.character(dias_entrenamiento), replace = TRUE)
  filas_b <- unlist(filas_por_dia[dias_b], use.names = FALSE)
  coef(lm(formula_lineal, data = entrenamiento_imp[filas_b, ]))[numericas]
})
alfa <- 0.05   # nivel de confianza del 95 %
tabla_coef <- tibble(
  variable      = numericas,
  coeficiente   = coef(modelo_ols)[numericas],
  ic_inf        = ic_ols[numericas, 1],
  ic_sup        = ic_ols[numericas, 2],
  ic_boot_inf   = map_dbl(boot_coef, quantile, probs = alfa / 2),
  ic_boot_sup   = map_dbl(boot_coef, quantile, probs = 1 - alfa / 2),
  p_valor       = summary(modelo_ols)$coefficients[numericas, 4],
  de_variable   = map_dbl(entrenamiento_imp[numericas], sd),
  efecto_1DE_MW = coeficiente * de_variable,
  efecto_pct_demanda_media = 100 * efecto_1DE_MW / mean(entrenamiento_imp$demanda)
)
print(tabla_coef, width = Inf)
write_csv(tabla_coef, "resultados/tabla_coeficientes_ols_ic.csv")

# Diagnóstico de residuos
png("resultados/fig03_diagnostico_ols.png", width = 1400, height = 1100, res = 150)
par(mfrow = c(2, 2))
plot(modelo_ols, which = c(1, 2, 3, 5), sub.caption = "", cex = 0.3)
dev.off()

residuos_ols <- residuals(modelo_ols)
png("resultados/fig04_autocorrelacion_residuos.png", width = 1200, height = 600, res = 150)
acf(residuos_ols, lag.max = 72, main = "Autocorrelación de los residuos OLS (rezagos en horas)")
dev.off()

distancia_cook <- cooks.distance(modelo_ols)
apalancamiento <- hatvalues(modelo_ols)
diagnostico_ols <- tibble(
  media_residuos      = mean(residuos_ols),
  de_residuos         = sd(residuos_ols),
  autocorr_lag1       = acf(residuos_ols, plot = FALSE)$acf[2],
  autocorr_lag24      = acf(residuos_ols, lag.max = 24, plot = FALSE)$acf[25],
  max_cook            = max(distancia_cook),
  n_cook_mayor_4_n    = sum(distancia_cook > 4 / length(distancia_cook)),
  n_cook_mayor_1      = sum(distancia_cook > 1),
  max_apalancamiento  = max(apalancamiento)
)
print(diagnostico_ols, width = Inf)
write_csv(diagnostico_ols, "resultados/tabla_diagnostico_ols.csv")

# Pruebas formales de los supuestos del OLS
set.seed(2026)
muestra_shapiro <- sample(residuos_ols, 5000)   # shapiro.test admite máx. 5.000 datos

bp  <- bptest(modelo_ols)                        # H0: varianza constante (homocedasticidad)
dw  <- dwtest(modelo_ols)                        # H0: sin autocorrelación de orden 1
lb  <- Box.test(residuos_ols, lag = 24, type = "Ljung-Box")  # H0: sin autocorrelación hasta lag 24
sw  <- shapiro.test(muestra_shapiro)             # H0: residuos normales

tabla_tests_ols <- tibble(
  supuesto  = c("Homocedasticidad", "Autocorrelación (lag 1)",
                "Autocorrelación (hasta lag 24)", "Normalidad de residuos"),
  prueba    = c("Breusch-Pagan", "Durbin-Watson", "Ljung-Box (24 rezagos)",
                "Shapiro-Wilk (muestra de 5.000)"),
  estadistico = c(unname(bp$statistic), unname(dw$statistic),
                  unname(lb$statistic), unname(sw$statistic)),
  p_valor   = c(bp$p.value, dw$p.value, lb$p.value, sw$p.value)
)
print(tabla_tests_ols, width = Inf)
write_csv(tabla_tests_ols, "resultados/tabla_tests_formales_ols.csv")

# Multicolinealidad: VIF_j = 1 / (1 - R²_j), donde R²_j viene de regresar
# la variable j contra las demás numéricas. VIF > 10 indica colinealidad severa.
vif_manual <- function(df, variables) {
  map_dbl(variables, function(v) {
    r2 <- summary(lm(reformulate(setdiff(variables, v), response = v), data = df))$r.squared
    1 / (1 - r2)
  })
}
tabla_vif <- tibble(variable = numericas, VIF = vif_manual(entrenamiento_imp, numericas)) %>%
  arrange(desc(VIF))
print(tabla_vif)
write_csv(tabla_vif, "resultados/tabla_vif.csv")



# 6. VALIDACIÓN CRUZADA TEMPORAL DE TODOS LOS MODELOS

# Para cada fold: imputar con datos del fold -> ajustar -> predecir el trimestre
# de validación. Los hiperparámetros (lambda, alpha, k, cp, mtry, profundidad,
# nº de árboles) se eligen con estos errores, nunca con el test.

cat("\nValidación cruzada temporal (", length(folds), "folds ):\n")
pred_cv <- bind_rows(
  validacion_temporal("Baseline",        ap_baseline),
  validacion_temporal("OLS",             ap_ols),
  validacion_temporal("Ridge/Lasso/EN",  ap_glmnet),
  validacion_temporal("GAM",             ap_gam),
  validacion_temporal("Árbol",           ap_arbol),
  validacion_temporal("Random Forest",   ap_rf),
  validacion_temporal("Boosting",        ap_gbm)
)
resumen_cv <- resumir_cv(pred_cv)

# Regularización: lambda.min y lambda.1se para cada alpha
resumen_glmnet <- resumen_cv %>%
  filter(modelo == "Ridge/Lasso/EN") %>%
  mutate(alpha  = as.numeric(str_match(config, "alpha=([0-9.]+)")[, 2]),
         lambda = as.numeric(str_match(config, "lambda=([0-9.e+-]+)")[, 2]))
lambdas_elegidos <- resumen_glmnet %>%
  group_by(alpha) %>%
  summarise(lambda_min = lambda[which.min(RMSE)],
            RMSE_min   = min(RMSE),
            lambda_1se = max(lambda[RMSE <= min(RMSE) + EE_RMSE[which.min(RMSE)]]),
            .groups = "drop") %>%
  mutate(metodo = recode(as.character(alpha), "0" = "Ridge", "0.5" = "Elastic-net", "1" = "Lasso"))
print(lambdas_elegidos)
write_csv(lambdas_elegidos, "resultados/tabla_lambdas_glmnet.csv")

g_glmnet <- resumen_glmnet %>%
  mutate(metodo = recode(as.character(alpha), "0" = "Ridge", "0.5" = "Elastic-net", "1" = "Lasso")) %>%
  ggplot(aes(log(lambda), RMSE, colour = metodo)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = RMSE - EE_RMSE, ymax = RMSE + EE_RMSE, fill = metodo),
              alpha = 0.15, colour = NA) +
  labs(title = "Error de validación cruzada según la penalización",
       subtitle = "Banda = ±1 error estándar entre folds",
       x = expression(log(lambda)), y = "RMSE de validación (MW)", colour = NULL, fill = NULL) +
  theme(legend.position = "bottom")
ggsave("resultados/fig05_cv_regularizacion.png", g_glmnet, width = 7, height = 4.5, dpi = 150)

#  Mejor configuración de cada modelo (mínimo RMSE promedio en CV)
mejores <- resumen_cv %>%
  group_by(modelo) %>% slice_min(RMSE, n = 1, with_ties = FALSE) %>% ungroup()

# Para la tabla se muestran ridge, lasso y elastic-net por separado (cada uno con su lambda.min)
mejores_glmnet <- resumen_glmnet %>%
  group_by(alpha) %>% slice_min(RMSE, n = 1, with_ties = FALSE) %>% ungroup()

# Combinación de modelos (averaging): promedio simple de los mejores modelos
#     individuales en CV. No se ajusta nada nuevo: se promedian sus predicciones.
componentes_promedio <- mejores %>%
  filter(modelo != "Baseline") %>%
  slice_min(RMSE, n = 3) %>%
  select(modelo, config)
cat("Componentes del averaging:", paste(componentes_promedio$modelo, collapse = " + "), "\n")
pred_promedio <- pred_cv %>%
  semi_join(componentes_promedio, by = c("modelo", "config")) %>%
  group_by(id, fold, y, no_laborable) %>%
  summarise(pred = mean(pred), .groups = "drop") %>%
  mutate(modelo = "Averaging",
         config = paste(componentes_promedio$modelo, collapse = " + "))
pred_cv <- bind_rows(pred_cv, pred_promedio)
mejores <- bind_rows(mejores, resumir_cv(pred_promedio))

# Correlación de errores entre componentes (el promedio ayuda si es < 1)
cor_errores <- pred_cv %>%
  semi_join(componentes_promedio, by = c("modelo", "config")) %>%
  mutate(error = y - pred) %>%
  select(id, modelo, error) %>%
  pivot_wider(names_from = modelo, values_from = error) %>%
  select(-id) %>% cor()
print(round(cor_errores, 3))



# 7. TABLA OBLIGATORIA DE VALIDACIÓN Y SELECCIÓN DEL SISTEMA

rmse_baseline     <- mejores$RMSE[mejores$modelo == "Baseline"]
rmse_sub_baseline <- mejores$RMSE_sub[mejores$modelo == "Baseline"]

tabla_validacion <- bind_rows(
  mejores %>% filter(modelo != "Ridge/Lasso/EN"),
  mejores_glmnet %>%
    mutate(modelo = recode(as.character(alpha), "0" = "Ridge", "0.5" = "Elastic-net", "1" = "Lasso")) %>%
    select(names(mejores))
) %>%
  left_join(catalogo %>% select(modelo, familia, complejidad), by = "modelo") %>%
  mutate(familia     = coalesce(familia, "Regularizado"),
         complejidad = coalesce(complejidad, 3),
         vs_baseline = sprintf("%+.1f%%", 100 * (RMSE / rmse_baseline - 1)),
         vs_baseline_sub = sprintf("%+.1f%%", 100 * (RMSE_sub / rmse_sub_baseline - 1))) %>%
  arrange(complejidad, RMSE) %>%
  transmute(Modelo = modelo, Familia = familia, Hiperparametros = config,
            RMSE = round(RMSE), `DE RMSE (folds)` = round(DE_RMSE), MAE = round(MAE),
            R2 = round(R2, 3), `RMSE subgrupo (no laborable)` = round(RMSE_sub),
            `vs baseline` = vs_baseline,
            `vs baseline (subgrupo)` = vs_baseline_sub)
print(tabla_validacion, width = Inf)
write_csv(tabla_validacion, "resultados/tabla_validacion.csv")

# Incertidumbre de la comparación con el baseline en validación: bootstrap por bloques de
# días sobre las predicciones de CV (el test no participa). Se remuestrean días completos
# porque las horas de un mismo día están correlacionadas. Para cada modelo (con su mejor
# configuración) se estima el cambio % del RMSE respecto al baseline, global y en el
# subgrupo (días no laborables), con IC del 95 %. Un IC que no contiene 0 indica que la
# diferencia con el baseline supera el ruido de muestreo. (Se usa el RMSE agrupado de
# todos los folds, que puede diferir levemente del promedio de RMSE por fold de la tabla.)
pred_tabla <- bind_rows(
  pred_cv %>%
    semi_join(mejores %>% filter(modelo != "Ridge/Lasso/EN") %>% select(modelo, config),
              by = c("modelo", "config")),
  pred_cv %>%
    filter(modelo == "Ridge/Lasso/EN") %>%
    semi_join(mejores_glmnet %>% select(config), by = "config") %>%
    mutate(modelo = recode(as.character(as.numeric(str_match(config, "alpha=([0-9.]+)")[, 2])),
                           "0" = "Ridge", "0.5" = "Elastic-net", "1" = "Lasso"))
) %>%
  mutate(fecha = entrenamiento$fecha[id], err2 = (y - pred)^2)

dias_cv <- pred_tabla %>%
  group_by(modelo, fecha) %>%
  summarise(sse     = sum(err2, na.rm = TRUE),
            n       = sum(!is.na(err2)),
            sse_sub = sum(err2[no_laborable == 1], na.rm = TRUE),
            n_sub   = sum(!is.na(err2) & no_laborable == 1),
            .groups = "drop") %>%
  arrange(modelo, fecha)

set.seed(2026)
n_dias_cv <- n_distinct(dias_cv$fecha)
indices_boot_cv <- replicate(1000, sample.int(n_dias_cv, replace = TRUE), simplify = FALSE)

cambio_rmse_pct <- function(m, base, ix, sub = FALSE) {
  if (sub) 100 * (sqrt(sum(m$sse_sub[ix]) / sum(m$n_sub[ix])) /
                    sqrt(sum(base$sse_sub[ix]) / sum(base$n_sub[ix])) - 1)
  else     100 * (sqrt(sum(m$sse[ix]) / sum(m$n[ix])) /
                    sqrt(sum(base$sse[ix]) / sum(base$n[ix])) - 1)
}
dias_baseline <- dias_cv %>% filter(modelo == "Baseline")
comparacion_cv_baseline <- map_dfr(setdiff(unique(dias_cv$modelo), "Baseline"), function(nombre) {
  dias_modelo <- dias_cv %>% filter(modelo == nombre)
  stopifnot(identical(dias_modelo$fecha, dias_baseline$fecha))
  todos <- seq_len(n_dias_cv)
  b_global <- map_dbl(indices_boot_cv, ~ cambio_rmse_pct(dias_modelo, dias_baseline, .x))
  b_sub    <- map_dbl(indices_boot_cv, ~ cambio_rmse_pct(dias_modelo, dias_baseline, .x, sub = TRUE))
  tibble(Modelo = nombre,
         `dif RMSE % (global)`    = cambio_rmse_pct(dias_modelo, dias_baseline, todos),
         `IC95 inf`               = quantile(b_global, alfa / 2),
         `IC95 sup`               = quantile(b_global, 1 - alfa / 2),
         `dif RMSE % (subgrupo)`  = cambio_rmse_pct(dias_modelo, dias_baseline, todos, sub = TRUE),
         `IC95 inf (sub)`         = quantile(b_sub, alfa / 2),
         `IC95 sup (sub)`         = quantile(b_sub, 1 - alfa / 2))
}) %>%
  mutate(`mejor que baseline (global)`   = `IC95 sup` < 0,
         `mejor que baseline (subgrupo)` = `IC95 sup (sub)` < 0,
         across(where(is.numeric), ~ round(.x, 1)))
print(comparacion_cv_baseline, width = Inf)
write_csv(comparacion_cv_baseline, "resultados/tabla_cv_vs_baseline_ic.csv")

# Selección con regla de parsimonia "1 error estándar": entre los modelos cuyo
# RMSE no supera (mejor RMSE + 1 EE), se elige el MÁS SIMPLE. Así sólo se acepta
# complejidad extra si mejora el error más allá del ruido entre folds.
candidatos_sel <- mejores %>%
  left_join(catalogo %>% select(modelo, complejidad), by = "modelo")
mejor_global <- candidatos_sel %>% slice_min(RMSE, n = 1, with_ties = FALSE)
umbral_1ee <- mejor_global$RMSE + mejor_global$EE_RMSE
seleccionado <- candidatos_sel %>%
  filter(RMSE <= umbral_1ee) %>%
  arrange(complejidad, RMSE) %>% slice(1)
cat(sprintf("\nMejor RMSE en CV: %s (%.0f MW, EE = %.0f). Umbral 1-EE: %.0f MW\n",
            mejor_global$modelo, mejor_global$RMSE, mejor_global$EE_RMSE, umbral_1ee))
cat(sprintf("SISTEMA SELECCIONADO (congelado): %s [%s]\n", seleccionado$modelo, seleccionado$config))

g_comp <- mejores %>%
  left_join(catalogo %>% select(modelo, complejidad), by = "modelo") %>%
  mutate(modelo = fct_reorder(modelo, -complejidad),
         elegido = modelo == seleccionado$modelo) %>%
  ggplot(aes(RMSE, modelo, colour = elegido)) +
  geom_vline(xintercept = umbral_1ee, linetype = "dashed", colour = "grey50") +
  geom_errorbar(aes(xmin = RMSE - EE_RMSE, xmax = RMSE + EE_RMSE), width = 0.25) +
  geom_point(size = 3) +
  scale_colour_manual(values = c(`FALSE` = "grey30", `TRUE` = "#D55E00"), guide = "none") +
  labs(title = "Comparación de modelos en validación cruzada temporal",
       subtitle = "Punto = RMSE promedio de 4 folds; barra = ±1 EE; línea = umbral regla 1-EE",
       x = "RMSE de validación (MW)", y = NULL)
ggsave("resultados/fig06_comparacion_modelos_cv.png", g_comp, width = 8, height = 4.5, dpi = 150)



# 8. INTERPRETACIÓN DE LOS MODELOS (ajustados con todo el entrenamiento)


# Coeficientes de lasso con lambda.1se (modelo más simple dentro de 1 EE)
x_entrenamiento <- matriz_x(entrenamiento_imp)
lasso_final <- glmnet(x_entrenamiento, entrenamiento_imp$demanda, alpha = 1,
                      lambda = grilla_lambda)
lambda_1se_lasso <- lambdas_elegidos$lambda_1se[lambdas_elegidos$alpha == 1]
coef_lasso <- as.matrix(coef(lasso_final, s = lambda_1se_lasso))
cat(sprintf("Lasso (lambda.1se = %.3g): %d de %d coeficientes distintos de cero\n",
            lambda_1se_lasso, sum(coef_lasso[-1, 1] != 0), nrow(coef_lasso) - 1))
png("resultados/fig07_trayectorias_lasso.png", width = 1200, height = 800, res = 150)
plot(lasso_final, xvar = "lambda", main = "Trayectorias de coeficientes (lasso)")
abline(v = log(lambda_1se_lasso), lty = 2)
dev.off()

# GAM: forma de los efectos no lineales
k_gam_elegido <- as.numeric(str_remove(mejores$config[mejores$modelo == "GAM"], "k="))
gam_final <- ajustar_gam(entrenamiento_imp, k_gam_elegido)
capture.output(summary(gam_final), file = "resultados/resumen_gam.txt")
png("resultados/fig08_efectos_gam.png", width = 1600, height = 1200, res = 150)
plot(gam_final, pages = 1, shade = TRUE, scale = 0)
dev.off()

# Árbol de regresión (podado para que sea legible)
arbol_visual <- rpart(formula_arboles, data = entrenamiento_imp, method = "anova",
                      control = rpart.control(cp = 0.01, xval = 0))
png("resultados/fig09_arbol_regresion.png", width = 1600, height = 1000, res = 150)
rpart.plot(arbol_visual, main = "Árbol de regresión (cp = 0.01)", roundint = FALSE)
dev.off()

# Random Forest: importancia de variables y gráfico de dependencia parcial (PDP)
mtry_elegido <- as.numeric(str_remove(mejores$config[mejores$modelo == "Random Forest"], "mtry="))
rf_final <- ranger(formula_arboles, data = entrenamiento_imp, num.trees = 300,
                   mtry = mtry_elegido, min.node.size = 5, importance = "permutation",
                   seed = 2026)
g_imp <- tibble(variable = names(rf_final$variable.importance),
                importancia = rf_final$variable.importance) %>%
  ggplot(aes(importancia, fct_reorder(variable, importancia))) + geom_col(fill = "#0072B2") +
  labs(title = "Importancia por permutación (Random Forest)",
       x = "Aumento del error cuadrático al permutar la variable", y = NULL)
ggsave("resultados/fig10_importancia_rf.png", g_imp, width = 7, height = 4.5, dpi = 150)

set.seed(2026)
muestra_pdp <- entrenamiento_imp[sample(nrow(entrenamiento_imp), 1000), c("demanda", predictores)]
pdp_hora <- pdp::partial(rf_final, pred.var = "hora", train = muestra_pdp, ice = TRUE,
                    pred.fun = function(object, newdata) predict(object, newdata)$predictions)
g_pdp <- ggplot(pdp_hora, aes(hora, yhat, group = yhat.id)) +
  geom_line(alpha = 0.05, colour = "grey40") +
  stat_summary(aes(group = 1), fun = mean, geom = "line", colour = "#D55E00", linewidth = 1.3) +
  labs(title = "Efecto de la hora del día (PDP e ICE, Random Forest)",
       subtitle = "Gris: curvas ICE (una por observación); naranja: PDP (promedio)",
       x = "Hora del día", y = "Demanda predicha (MW)")
ggsave("resultados/fig11_pdp_ice_hora.png", g_pdp, width = 7, height = 4.5, dpi = 150)


# 9.APERTURA ÚNICA DEL TEST (2018): sistema congelado vs baseline

stopifnot(!exists("prueba"))   # garantía: el test no se ha creado antes de este punto
prueba <- datos %>% filter(fecha > fecha_corte)
imp_prueba <- imputar_mediana(entrenamiento, prueba)   # medianas del ENTRENAMIENTO
prueba_imp <- imp_prueba$nuevos

# Reajusta un modelo con todo el entrenamiento y predice el test con la configuración elegida
predecir_test <- function(nombre, config) {
  funcion <- get(catalogo$funcion[catalogo$modelo == nombre])
  pred <- switch(nombre,
    "Ridge/Lasso/EN" = funcion(entrenamiento_imp, prueba_imp,
                               alphas = as.numeric(str_match(config, "alpha=([0-9.]+)")[, 2])),
    "GAM"            = funcion(entrenamiento_imp, prueba_imp, ks = as.numeric(str_remove(config, "k="))),
    "Árbol"          = funcion(entrenamiento_imp, prueba_imp, cps = as.numeric(str_remove(config, "cp="))),
    "Random Forest"  = funcion(entrenamiento_imp, prueba_imp, mtrys = as.numeric(str_remove(config, "mtry="))),
    "Boosting"       = funcion(entrenamiento_imp, prueba_imp,
                               profundidades = as.numeric(str_match(config, "prof=([0-9]+)")[, 2]),
                               shrinkages    = as.numeric(str_match(config, "shr=([0-9.]+)")[, 2])),
    funcion(entrenamiento_imp, prueba_imp))
  pred[, config]
}

if (seleccionado$modelo == "Averaging") {
  pred_sistema <- map(seq_len(nrow(componentes_promedio)), function(i)
    predecir_test(componentes_promedio$modelo[i], componentes_promedio$config[i])) %>%
    reduce(`+`) / nrow(componentes_promedio)
} else {
  pred_sistema <- predecir_test(seleccionado$modelo, seleccionado$config)
}
pred_baseline_test <- predecir_test("Baseline", "promedio_hora_dia")

resultados_test <- prueba_imp %>%
  select(fecha_local, fecha, hora, no_laborable, demanda, pronostico_tso) %>%
  mutate(pred_sistema = pred_sistema, pred_baseline = pred_baseline_test)

# Intervalo de predicción empírico (95%): cuantiles de los errores de validación
#     del sistema seleccionado en la CV (no usa el test).
errores_cv_sistema <- pred_cv %>%
  filter(modelo == seleccionado$modelo, config == seleccionado$config) %>%
  mutate(error = y - pred) %>% pull(error)
cuantiles_error <- quantile(errores_cv_sistema, c(alfa / 2, 1 - alfa / 2))
resultados_test <- resultados_test %>%
  mutate(lim_inf = pred_sistema + cuantiles_error[1],
         lim_sup = pred_sistema + cuantiles_error[2])

# Métricas en el test: global y subgrupo (días no laborables)
evaluar <- function(df, nombre_pred, etiqueta) {
  df <- df %>% filter(!is.na(.data[[nombre_pred]]))
  bind_rows(
    metricas(df$demanda, df[[nombre_pred]]) %>% mutate(grupo = "Todas las horas"),
    df %>% filter(no_laborable == 1) %>%
      { metricas(.$demanda, .[[nombre_pred]]) } %>% mutate(grupo = "Días no laborables")
  ) %>% mutate(modelo = etiqueta, .before = 1)
}
tabla_test <- bind_rows(
  evaluar(resultados_test, "pred_baseline", "Baseline (promedio hora x día)"),
  evaluar(resultados_test, "pred_sistema", paste("Sistema seleccionado:", seleccionado$modelo)),
  evaluar(resultados_test, "pronostico_tso", "Referencia: pronóstico oficial TSO")
) %>% mutate(across(c(RMSE, MAE), round), R2 = round(R2, 3))
print(tabla_test, n = Inf)
write_csv(tabla_test, "resultados/tabla_test.csv")

# Incertidumbre: bootstrap por bloques de días del RMSE y de la diferencia vs baseline
set.seed(2026)
filas_dia_test <- split(seq_len(nrow(resultados_test)), resultados_test$fecha)
boot_test <- map_dfr(seq_len(1000), function(b) {   # 1.000 remuestreos
  filas_b <- unlist(filas_dia_test[sample(names(filas_dia_test), replace = TRUE)], use.names = FALSE)
  d <- resultados_test[filas_b, ]
  d_sub <- d[d$no_laborable == 1, ]
  tibble(rmse_sistema      = sqrt(mean((d$demanda - d$pred_sistema)^2)),
         rmse_baseline     = sqrt(mean((d$demanda - d$pred_baseline)^2)),
         rmse_sub_sistema  = sqrt(mean((d_sub$demanda - d_sub$pred_sistema)^2)),
         rmse_sub_baseline = sqrt(mean((d_sub$demanda - d_sub$pred_baseline)^2)))
}) %>% mutate(diferencia     = rmse_baseline - rmse_sistema,
              mejora_pct     = 100 * diferencia / rmse_baseline,
              mejora_sub_pct = 100 * (rmse_sub_baseline - rmse_sub_sistema) / rmse_sub_baseline)
ic_test <- boot_test %>%
  summarise(across(everything(), list(
    estimacion = ~ mean(.x),
    ic_inf     = ~ quantile(.x, alfa / 2),
    ic_sup     = ~ quantile(.x, 1 - alfa / 2)))) %>%
  pivot_longer(everything(), names_to = c("medida", ".value"),
               names_pattern = "(.*)_(estimacion|ic_inf|ic_sup)")
cobertura <- resultados_test %>%
  summarise(cobertura_intervalo = mean(demanda >= lim_inf & demanda <= lim_sup),
            ancho_promedio_MW   = mean(lim_sup - lim_inf))
print(ic_test)
print(cobertura)
write_csv(ic_test, "resultados/tabla_test_bootstrap_ic.csv")
write_csv(cobertura, "resultados/tabla_test_intervalo_prediccion.csv")

#Gráficos del test
semana_ejemplo <- resultados_test %>%
  filter(fecha >= as.Date("2018-03-05"), fecha <= as.Date("2018-03-11"))
g_semana <- ggplot(semana_ejemplo, aes(fecha_local)) +
  geom_ribbon(aes(ymin = lim_inf, ymax = lim_sup), fill = "#D55E00", alpha = 0.2) +
  geom_line(aes(y = demanda, colour = "Real"), linewidth = 0.9) +
  geom_line(aes(y = pred_sistema, colour = "Sistema seleccionado"), linewidth = 0.7) +
  geom_line(aes(y = pred_baseline, colour = "Baseline"), linewidth = 0.6, linetype = "dashed") +
  scale_colour_manual(values = c(Real = "black", `Sistema seleccionado` = "#D55E00",
                                 Baseline = "#0072B2")) +
  labs(title = "Test 2018: semana del 5 al 11 de marzo",
       subtitle = "Banda: intervalo de predicción empírico del 95%",
       x = NULL, y = "Demanda (MW)", colour = NULL) +
  theme(legend.position = "bottom")
ggsave("resultados/fig12_test_semana.png", g_semana, width = 9, height = 4.5, dpi = 150)

g_mes <- resultados_test %>%
  mutate(mes = month(fecha, label = TRUE)) %>%
  group_by(mes) %>%
  summarise(Sistema  = sqrt(mean((demanda - pred_sistema)^2)),
            Baseline = sqrt(mean((demanda - pred_baseline)^2)), .groups = "drop") %>%
  pivot_longer(-mes, names_to = "modelo", values_to = "RMSE") %>%
  ggplot(aes(mes, RMSE, fill = modelo)) + geom_col(position = "dodge") +
  scale_fill_manual(values = c(Sistema = "#D55E00", Baseline = "#0072B2")) +
  labs(title = "Error del test por mes (2018)", x = NULL, y = "RMSE (MW)", fill = NULL) +
  theme(legend.position = "bottom")
ggsave("resultados/fig13_test_error_mensual.png", g_mes, width = 8, height = 4, dpi = 150)



# 10.VERIFICACIONES DE INTEGRIDAD Y REGISTRO

verificaciones <- c(
  "Test posterior a todo el entrenamiento" =
    max(entrenamiento$fecha) < min(prueba$fecha),
  "Sin filas compartidas entre entrenamiento y test" =
    length(intersect(entrenamiento$fecha_utc, prueba$fecha_utc)) == 0,
  "Cada fold valida DESPUÉS de lo que entrena (con purga)" =
    all(map_lgl(folds, ~ max(entrenamiento$fecha[.x$entrenar]) <=
                          min(entrenamiento$fecha[.x$validar]) - 2)),   # 1 día de purga
  "Folds de validación sin superposición" =
    !any(duplicated(unlist(map(folds, "validar")))),
  "Ningún predictor usa información de la misma hora" =
    !any(c("demanda", "pronostico_tso", "precio_da", "gen_solar", "gen_eolica", "gen_gas")
         %in% predictores),
  "Rezago de 24 h correcto (comparado por fecha)" = {
    control <- datos_completos %>% select(fecha_utc, demanda_lag24) %>%
      inner_join(datos_completos %>% transmute(fecha_utc = fecha_utc + hours(24), demanda_real = demanda),
                 by = "fecha_utc")
    isTRUE(all.equal(control$demanda_lag24, control$demanda_real))
  },
  "Serie horaria regular (sin huecos ni duplicados)" =
    all(diff(as.numeric(datos_completos$fecha_utc)) == 3600),
  "Medianas de imputación del test calculadas con el entrenamiento" =
    isTRUE(all.equal(map_dbl(entrenamiento[predictores], median, na.rm = TRUE),
                     map_dbl(imp_prueba$ajuste[predictores], median)))
)
tabla_verificaciones <- tibble(verificacion = names(verificaciones), resultado = verificaciones)
print(tabla_verificaciones)
write_csv(tabla_verificaciones, "resultados/tabla_verificaciones.csv")
if (!all(verificaciones)) stop("Verificación de integridad fallida: ",
                                paste(names(verificaciones)[!verificaciones], collapse = "; "))

# Registro de la ejecución y del entorno (reproducibilidad): fecha, semilla,
# modelo elegido y versiones de R y de los paquetes
sink("resultados/registro_configuracion.txt")
cat("Ejecución:", format(Sys.time()), "\nSemilla: 2026\n")
cat("Entrenamiento hasta:", format(fecha_corte), "| Test: 2018\n")
cat("Sistema seleccionado:", seleccionado$modelo, "|", seleccionado$config, "\n\n")
print(sessionInfo())
sink()

cat("\n================ RESUMEN FINAL ================\n")
cat(sprintf("Sistema seleccionado: %s (%s)\n", seleccionado$modelo, seleccionado$config))
cat(sprintf("RMSE test sistema : %.0f MW\n", tabla_test$RMSE[3]))
cat(sprintf("RMSE test baseline: %.0f MW\n", tabla_test$RMSE[1]))
cat("Resultados guardados en la carpeta 'resultados'\n")
