Fedrated Unlearning - PFLIB fork
in order to run the client wise unlearning model using the FedAvg algo on MNIST using CNN architecture
first 50 clients are used to train the model remainder are forgotten

```
cd system/
python main.py --learn False --learn_count 50
```