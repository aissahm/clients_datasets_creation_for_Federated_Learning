## Creates clients datasets as non-iid for Federated Learning

Note: this works for classification tasks only. 


How it works:

Given a dataset X and Y, the dataloader returns an object of elements with client_id and a list of indexes of samples to pick from dataset X and Y.

Main parameters: 

- (1) the number of clients in the Federated Learning environment,
- (2) the share of the main class contained in each client sample.

Example: 

loadMNISTdata.py contains an example on how to create 100 client data samples. 

Each client dataset contains 50% of samples from one class, and 50% of samples from other classes equally picked.

