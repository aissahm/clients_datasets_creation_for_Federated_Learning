import random
import numpy as np

#return an object containing the indexes of samples for every client from dataset X, Y 
#"client_i": [index1(i), index2(i), index3(i), ...] containing indexes picked randomly
# according to non-iid level alpha_dis representing % of main class in client dataset
def returnClientDatasetsNonIIDdata(Y, num_clients, alpha_dis = 0.5):
  clients_data_obj = {}
  N = Y.shape[0]
  unique, counts = np.unique(Y, return_counts=True)

  clustered_dataset = {}
  num_classes = len(unique)

  num_clients_per_cluster = num_clients//num_classes
  client_sample_size = N//num_clients

  classes_list = []
  #get the indexes of each class from Y data
  for label in unique:
    class_indexes = list(np.where(Y == label)[0])
    clustered_dataset[label] = class_indexes
    classes_list.extend([label] * num_clients_per_cluster)
  
  #creating a list containing random main class for each client in FL
  randomized_class_list = random.sample(classes_list, k=num_classes * num_clients_per_cluster)

  dominant_class_num_samples = int(client_sample_size * alpha_dis)
  remaining_classes_num_samples = client_sample_size - dominant_class_num_samples
  
  for i in range(num_clients):
    
    #dominant class
    class_pick = randomized_class_list[i]
    class_num_elements = len(clustered_dataset[class_pick])

    sample_size = dominant_class_num_samples

    #in case, there are fewer elements to chose from remaining dataset Y
    if class_num_elements < sample_size:
      sample_size = class_num_elements

    indexes = list(np.random.choice(range(class_num_elements), size=sample_size, replace=False))
    
    client_data_indexes = np.array(clustered_dataset[class_pick])[indexes]
    
    new_label_indexes = list(filter(lambda elem: clustered_dataset[class_pick].index(elem) not in indexes, clustered_dataset[class_pick]))
    clustered_dataset[class_pick] = new_label_indexes

    other_classes_num_samples = int(remaining_classes_num_samples / (num_classes - 1.0))

    #randomly picking samples from remaining classes
    for label in unique:
      if label != class_pick:
        class_num_elements = len(clustered_dataset[label])

        sample_size = other_classes_num_samples
        if class_num_elements < sample_size:
          sample_size = class_num_elements

        indexes = list(np.random.choice(range(class_num_elements), size=sample_size, replace=False))

        client_data_indexes = np.append(client_data_indexes, np.array(clustered_dataset[label])[indexes], axis=0).astype(int)

        #removing the indexes chosen for the client (sampling without replacement)
        new_label_indexes = list(filter(lambda elem: clustered_dataset[label].index(elem) not in indexes, clustered_dataset[label]))
        clustered_dataset[label] = new_label_indexes
        
    clients_data_obj[i] = {"main_class": class_pick, "indexes": client_data_indexes }
  
  indexes_list = list(range(0, N))

  for client in clients_data_obj:
    client_samples = clients_data_obj[client]["indexes"].shape[0]
    if client_samples != client_sample_size:
      num_samples_to_add = client_sample_size - client_samples
      indexes_to_add_list = indexes_list[0:num_samples_to_add]
      current_sample = clients_data_obj[client]["indexes"].tolist()
      new_client_sample = np.array(current_sample + indexes_to_add_list )
      clients_data_obj[client]["indexes"] = new_client_sample
      
  return clients_data_obj

#given the dataset X, Y, the object with indexes for every client, returns the dataset of client identified with its client_id
def returnClientDataset(client_id, clients_data_obj, x, y):
  dataset_indexes = np.array(clients_data_obj[client_id]["indexes"])
  return [x[dataset_indexes], y[dataset_indexes]]
