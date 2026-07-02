# NYC Taxi · Previsão de Gorjeta Alta — Pipeline MLOps na AWS

Pipeline de ML de ponta a ponta na AWS que prevê se uma corrida de táxi em Nova York
vai gerar **gorjeta alta** (> 20% do valor da corrida), a partir dos dados públicos
do *NYC TLC Trip Record Data*.

O projeto cobre o ciclo completo: **ETL com AWS Glue → Feature Store no SageMaker AI →
treino e avaliação (XGBoost) → deploy serverless e inferência em tempo real.**

---

## Arquitetura

```
┌─────────────┐   AWS CLI   ┌──────────────┐   AWS Glue    ┌──────────────────┐
│ s3://nyc-tlc│ ──────────▶ │ S3  raw/     │ ───────────▶  │ S3  processed/   │
│ (open data) │   cp/sync   │ (parquet)    │  ETL + label  │ (features+label) │
└─────────────┘             └──────────────┘               └────────┬─────────┘
                                                                     │ ingest
                                                                     ▼
                                                        ┌────────────────────────┐
                                                        │ SageMaker Feature Store│
                                                        │  online  +  offline    │
                                                        └─────┬───────────────┬──┘
                                          Athena (offline)    │               │ get_record (online)
                                                               ▼               ▼
                                                   ┌──────────────────┐   ┌──────────────────┐
                                                   │ XGBoost training │   │ Serverless       │
                                                   │ + Batch Transform│   │ Endpoint (infer.)│
                                                   │ (AUC, ROC, CM)   │   └──────────────────┘
                                                   └──────────────────┘
```

## Dataset

[NYC TLC Trip Record Data](https://registry.opendata.aws/nyc-tlc-trip-records-pds/)
— Registry of Open Data on AWS, formato Parquet, bucket `s3://nyc-tlc` (us-east-1).

### Decisões de dados que valem destaque
- **Só cartão de crédito (`payment_type = 1`).** No dataset do TLC, gorjetas em
  dinheiro não são registradas.
- **Sem vazamento de label.** `total_amount` e `tip_amount` **não** são usados como
  feature. `fare_amount` (pré-gorjeta) é seguro.

---

## Como rodar

### Pré-requisitos
- Conta AWS, tudo em **us-east-1**
- SageMaker Studio com uma IAM Role com acesso a S3, Glue, Athena e Feature Store
- Um bucket S3 com as zonas `raw/`, `processed/`, `feature-store/`

### Passos
1. **Ingestão** — copie um mês do dataset público para a sua zona `raw/`:
   ```bash
   aws s3 cp s3://nyc-tlc/trip\ data/yellow_tripdata_2025-01.parquet \
             s3://<seu-bucket>/raw/yellow/yellow_tripdata_2025-01.parquet
   ```
2. **ETL (Glue)** — crie um Glue Job apontando para `glue/etl_job.py` com os parâmetros:
   - `--raw_path s3://<seu-bucket>/raw/yellow/`
   - `--processed_path s3://<seu-bucket>/processed/yellow/`
   - `--high_tip_threshold 0.20`
3. **Notebooks** (no SageMaker Studio, na ordem):
   - `notebooks/feature_store.ipynb` — cria o Feature Group e ingere as features
   - `notebooks/rain_eval.ipynb` — treina o XGBoost e avalia (AUC, ROC, matriz de confusão)


> Edite as variáveis de config no topo do notebook (`BUCKET`, `REGION`, etc.).

---

## Serviços AWS por etapa

| Etapa            | Serviços |
|------------------|----------|
| Ingestão         | Amazon S3, AWS CLI |
| ETL              | AWS Glue (Job, Data Catalog, Crawler opcional), Athena |
| Feature storage  | SageMaker AI Feature Store (online + offline), S3, Glue Data Catalog |
| Treino           | SageMaker , XGBoost built-in |
| Avaliação        | SageMaker Batch Transform, scikit-learn |
| Deploy / Inferência | SageMaker Serverless Endpoint, Feature Store online |

---

## Resultado esperado

- Pipeline funcional ponta a ponta, com features versionadas no Feature Store
- Modelo XGBoost com AUC tipicamente na faixa de **~0,70–0,85** (varia com o limiar do label)
- Endpoint serverless retornando a probabilidade de gorjeta alta por corrida
- Gráficos de avaliação salvos em `diagrams/eval.png`

---

## 📁 Estrutura

```
aws-ny-taxi/
├── README.md
├── diagrams/                 # eval.png (gerado pelo notebook)
├── glue/
│   └── etl_job.py            # Glue ETL (limpeza + feature engineering + label)
├── notebooks/
│   ├── feature_store.ipynb
│   ├── train_eval.ipynb
├── requirements.txt
└── .gitignore
```

## 📝 Licença

Código sob MIT. O dataset segue os [termos de uso do NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).