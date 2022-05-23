# Machine_Learning_project
## Codebase for CS_251 ML course
### Starting the application
To have the model for starting up the app, run the naive bayes training script on augmented data in the model train folder. Then, save the model and its vectorizer with the joblib library like in the corresponding notebook. Finally, move the saved model and vectorizer files to the model_app folder.
```bash
streamlit run model_app/app.py
```
