```text
                    MLOps Platform
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   Data Lifecycle                    Model Lifecycle
        │                                 │
 Data Validation                       Training
        │                                 │
 Data Preparation                    Validation
        │                                 │
 Feature Engineering                Model Registry
        │                                 │
        └──────────────┬──────────────────┘
                       │
                 Release Lifecycle
                       │
                Model Image Builder
                       │
                 Container Image
                       │
                Container Registry
                       │
                Kubernetes Deploy
                       │
                   Monitoring
                       │
                Retraining Decision
                       │
                       └──────→ Training
```



``` text
Data
 ↓
Data Validation
 ↓
Training
 ↓
Model Validation
 ↓
Registered Model
      ↓
Build inference image
      ↓
Container image
      ↓
Push image
      ↓
Container Registry
      ↓
Deploy image to Kubernetes
      ↓
Monitor deployment/model
```
