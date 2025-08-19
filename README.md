<div align="center">

<img src="assets/logo.jpg" alt="AlgoML-Collection Logo" width="400"/>

# 🤖 AlgoML-Collection: Machine Learning Algorithms Collection

</div>

<div align="center">

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Algorithms-blue)
![Data Science](https://img.shields.io/badge/Data%20Science-Analytics-green)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Library-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

*A comprehensive implementation of fundamental machine learning algorithms using Python and scikit-learn. This repository demonstrates practical applications of supervised and unsupervised learning techniques across various real-world datasets.*

</div>

## 📊 Algorithms Implemented

### 🎯 Supervised Learning

#### 1. **Support Vector Machine (SVM)**
- **Dataset**: Breast cancer cell classification
- **Implementation**: Multiple kernel comparison (RBF, Polynomial, Sigmoid)
- **Evaluation**: Confusion matrix, F1-score, Jaccard index, Log loss
- **Use Case**: Binary classification for medical diagnosis

#### 2. **Logistic Regression**
- **Dataset**: Customer churn prediction
- **Features**: Customer demographics and usage patterns
- **Evaluation**: Precision, recall, accuracy, classification report
- **Use Case**: Business analytics and customer retention

#### 3. **K-Nearest Neighbors (KNN)**
- **Dataset**: Telecommunications customer segmentation
- **Implementation**: Optimal K selection with cross-validation
- **Visualization**: Accuracy vs. K-value plots with confidence intervals
- **Use Case**: Customer categorization and targeted marketing

#### 4. **Decision Trees**
- **Dataset**: Drug prescription classification
- **Features**: Patient demographics and medical indicators
- **Implementation**: Entropy-based splitting with pruning
- **Use Case**: Medical decision support systems

### 🔍 Unsupervised Learning

#### 5. **K-Means Clustering**
- **Implementation**: Synthetic and real-world data clustering
- **Features**: Centroid visualization and cluster optimization
- **Applications**: Customer segmentation and pattern recognition
- **Datasets**: Synthetic blobs and customer segmentation data

#### 6. **Hierarchical Clustering**
- **Methods**: Agglomerative clustering with multiple linkage criteria
- **Visualization**: Dendrograms and cluster trees
- **Comparison**: Different distance metrics and standardization effects
- **Use Case**: Taxonomy creation and data organization

#### 7. **DBSCAN (Density-Based Clustering)**
- **Implementation**: Density-based spatial clustering with noise detection
- **Comparison**: Performance analysis against K-Means
- **Parameters**: Epsilon and minimum samples optimization
- **Use Case**: Anomaly detection and irregular cluster shapes

## 🛠️ Technical Stack

- **Python 3.x**
- **scikit-learn** - Machine learning algorithms
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **scipy** - Scientific computing

## 📁 Repository Structure

```
ML-Assignments-master/
├── SVM/                    # Support Vector Machine
│   ├── SVM.py
│   └── cell_samples.csv
├── LogisticReg/           # Logistic Regression
│   ├── Logistic_Reg.py
│   └── ChurnData.csv
├── K-NearestNeigh/        # K-Nearest Neighbors
│   ├── K-NearestNeigh.py
│   └── teleCust1000t.csv
├── DecisionTrees/         # Decision Trees
│   ├── decisionTrees.py
│   ├── drug200.csv
│   └── newData.csv
├── K-Means/               # K-Means Clustering
│   ├── K-Means-1.py
│   ├── K-Means-2.py
│   └── Cust_Segmentation.csv
├── Hierarchical/          # Hierarchical Clustering
│   ├── Hierarchical-1.py
│   ├── Hierarchical-2.py
│   └── cars_clus.csv
└── DBSCAN/               # DBSCAN Clustering
    ├── DBSCAN.py
    ├── Weather_station_clustring.py
    └── weather-stations20140101-20141231.csv
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

### Running the Examples

Each algorithm is self-contained and can be run independently:

```bash
# Navigate to specific algorithm directory
cd SVM/
python SVM.py

# Or run from root directory
python SVM/SVM.py
```

## 📈 Key Features

- **Comprehensive Evaluation**: Each implementation includes multiple evaluation metrics
- **Data Preprocessing**: Proper feature scaling and encoding techniques
- **Visualization**: Clear plots for model performance and data distribution
- **Real-world Datasets**: Practical applications across healthcare, business, and telecommunications
- **Comparative Analysis**: Multiple algorithms for similar problems
- **Parameter Optimization**: Grid search and cross-validation techniques

## 🎯 Learning Outcomes

This repository demonstrates:

- **Algorithm Selection**: Choosing appropriate algorithms for different problem types
- **Data Preprocessing**: Handling categorical variables, scaling, and cleaning
- **Model Evaluation**: Using appropriate metrics for classification and clustering
- **Hyperparameter Tuning**: Optimizing model parameters for better performance
- **Visualization**: Creating meaningful plots to understand data and results

## 📊 Performance Highlights

- **SVM**: Achieves high accuracy in cancer cell classification with RBF kernel
- **Logistic Regression**: Effective customer churn prediction with regularization
- **KNN**: Optimal performance with K=4 for customer segmentation
- **Decision Trees**: Clear decision boundaries for drug prescription
- **Clustering**: Successful pattern identification in customer and geographical data

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for:
- Algorithm improvements
- Additional evaluation metrics
- New datasets
- Documentation enhancements

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**⭐ If you found this helpful, please give it a star! ⭐**

</div>