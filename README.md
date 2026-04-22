# Credit Risk Scoring ML

## Descripción
Sistema de scoring crediticio para originación de crédito,
desarrollado con Machine Learning para reducir cartera morosa.

## Objetivo
Predecir la probabilidad de incumplimiento (PD) de solicitantes
de crédito, permitiendo decisiones de aprobación/rechazo
basadas en riesgo real y no en criterios subjetivos.

## Dataset
Home Credit Default Risk (Kaggle)
- 300,000+ solicitudes de crédito reales
- Variable objetivo: incumplimiento en primeros 12 meses

## Metodología
1. Análisis exploratorio de datos (EDA)
2. Feature Engineering (WoE, DTI, variables de comportamiento)
3. Modelado: Scorecard Logístico + LightGBM calibrado
4. Validación: Out-of-time, tabla de deciles, Expected Loss
5. Producción: MLflow + Azure ML

## Stack Tecnológico
- Python, pandas, numpy, scikit-learn
- LightGBM, MLflow
- Azure ML (despliegue)

## Resultados
*(Se actualizará conforme avance el proyecto)*

## Autor
Marín Serrato Barrios
Actuario | Analytics & BI | Riesgo Crediticio