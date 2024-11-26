Fedrated Unlearning - PFLIB fork
in order to run the client wise unlearning model using the FedAvg algo on MNIST using CNN architecture
first 50 clients are used to train the model remainder are forgotten

## Data Generation Procedure
For PACS, where data spans multiple domains, we allocate clients based on domain-specific segmentation, meaning that each client is assigned to a specific domain. This approach allows FOUL to focus on selectively unlearning the influence of particular domains. To achieve an IID and balanced data distribution among clients, we recommend implementing the following procedure:
```
python generate_PACS1.py iid balance -
```
To Run FedAvg Algorithm:
```
cd system/
python main.py --learn False --learn_count 50
```

```
python main.py --learn unlearn --learn_count 50 --algo FOUL 
```
