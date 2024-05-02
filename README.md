### Результаты
В моем ноутбуке vk_train.ipynb я сделал сравнение трех моделей    


## Шаги выполнения задачи

1) Препроцессинг текста:
  - Нижний регистр
  - Токенезация
  - Удаление стоп слов
  - Лемматизация
2) Создал эмбеддинги TF-IDF
3) Сделал GridSearch по параметрам для моделей SVM, LogReg, NB
4) Выбрал лучшую модель и обучил на Train
5) Сделал предсказание на Test

## Метрики:
### SVM model:  
Accuracy: 0.9404176904176904    
Precision: 0.8910191725529768    
Recall: 0.9112487100103199    
F1 score: 0.9010204081632653    
ROC AUC score: 0.9320


### Logistic Regression model:    
Accuracy: 0.9161547911547911    
Precision: 0.8486973947895792    
Recall: 0.8740970072239422    
F1 score: 0.8612099644128114    
ROC AUC score: 0.9040    

### MultinomialNB model:    
Accuracy: 0.9299754299754299    
Precision: 0.8614634146341463    
Recall: 0.9112487100103199    
F1 score: 0.8856569709127382    
ROC AUC score: 0.9246    


# *В результате лучшие метрики показала модель SVC. Её я и выбрал.*
Результат в файле result.csv
