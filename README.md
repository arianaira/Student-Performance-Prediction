# Student Performance Prediction

## Abstract
This study focuses on predicting student performance as a **regression task**. After preprocessing the dataset, a baseline **MLP model** with the **Adam optimizer** and **L1 loss** was built. However, this model suffered from **overfitting**, prompting a systematic exploration of **regularization techniques**, **activation functions**, and **optimizers** to enhance performance and find the most optimal configuration.  
In this study, several mitigation techniques were explored, including:  
- **Regularization:** L2 regularization, Dropout, and Batch Normalization    
- **Activation Functions:** ReLU, Sigmoid, Tanh, Leaky ReLU, GELU, and Scaled Tanh    
- **Optimizers:** Adam and SGD  

The goal was to identify the most effective combination of these elements to achieve high predictive accuracy.  

---

## Data Preparation  

**Dataset shape:** `(395, 33)`  
**Target variable:** `G3` (final grade)  

### Features  
The dataset contained demographic, social, and academic attributes such as:  
age, Medu, Fedu, traveltime, studytime, failures, schoolsup, famsup,
paid, activities, nursery, higher, internet, romantic, famrel, freetime,
goout, Dalc, Walc, health, absences, G1, G2, G3, sex_M, guardian_mother,
reason_home, Fjob_teacher, Mjob_services, etc.  

### Preprocessing Steps  
1. **Missing Values:** None found.   
2. **Encoding:**  
   - Binary categorical (`yes`/`no`) → 0/1  
   - Other categorical variables → One-hot encoding  
3. **Normalization:** Applied `MinMaxScaler` to numerical columns.  
4. **Outlier Removal:** Removed outliers in `G2` and `G3` from the training set.  
5. **Train/Test Split:**  
   - Train: 80% (316 samples)  
   - Test: 20% (79 samples)  
After encoding, the dataset had **42 features**.  
---

##  Model Architectures  

### Baseline Model  
MLP(  
  (0): Linear(41, 64)  
  (1): ReLU()  
  (2): Linear(64, 128)
  (3): ReLU()  
  (4): Linear(128, 256)  
  (5): ReLU()  
  (6): Linear(256, 1)  
)  
Optimizer: Adam  
Loss Function: L1 Loss  

Test Results:  
MSE: 0.0226  
MAE: 0.1142  
R²: 0.5934  

| Model Variant                       | Key Modification             | MSE        | MAE        | R²        | Observation             |
| ----------------------------------- | ---------------------------- | ---------- | ---------- | --------- | ----------------------- |
| **MLP + L2**                        | L2 regularization            | 0.0223     | 0.1138     | 0.599     | Minor improvement       |
| **MLP + Dropout (p=0.1)**           | Dropout regularization       | 0.0206     | 0.1123     | 0.630     | Significant improvement |
| **MLP + Dropout (extra layer)**     | 512-unit layer + dropout=0.2 | 0.0237     | 0.1207     | 0.573     | Worse performance       |
| **MLP + Tanh**                      | Activation = Tanh            | 0.0155     | 0.0838     | 0.720     | Major improvement       |
| **MLP + Sigmoid**                   | Activation = Sigmoid         | 0.0208     | 0.0868     | 0.627     | Moderate improvement    |
| **MLP + Leaky ReLU**                | Activation = Leaky ReLU      | 0.0217     | 0.1142     | 0.609     | Similar to ReLU         |
| **MLP + SGD**                       | Optimizer = SGD              | 0.0144     | 0.0752     | 0.742     | Better than Adam        |
| **MLP + BatchNorm**                 | Added normalization          | 0.0240     | 0.1150     | 0.569     | Underfitting            |
| **MLP + GELU**                      | Activation = GELU            | 0.0171     | 0.0853     | 0.693     | Moderate improvement    |
| **MLP + Scaled Tanh (1000 epochs)** | Best configuration           | **0.0138** | **0.0786** | **0.753** | Optimal model         |


## Best Model Configuration  
Architecture:  
MLP_tanh_scaled(  
  (0): Linear(41, 64)  
  (1): ScaledTanh()  
  (2): Dropout(0.1)  
  (3): Linear(64, 128)  
  (4): ScaledTanh()  
  (5): Dropout(0.1)  
  (6): Linear(128, 256)  
  (7): ScaledTanh()  
  (8): Dropout(0.1)  
  (9): Linear(256, 1)  
)  
Optimizer: SGD  
Activation: Scaled Tanh   
Dropout Probability: 0.1  
Epochs: 1000  

Performance:  
Mean Squared Error: 0.01375  
Mean Absolute Error: 0.0786  
R² Score: 0.7526  

## Conclusion  
The optimal configuration for student performance prediction was achieved using:  
- Scaled Tanh activation function  
- Dropout (p=0.1) regularization  
- SGD optimizer  
- 1000 training epochs  
This configuration achieved the best test R² score of 0.7526, indicating strong predictive capability and generalization.   
