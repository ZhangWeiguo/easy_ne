### Network Embedding/ Network Representation Learning
    * Step
        * Get some train data like /data/${data_name}
            * Must: nodes.csv (show all the nodes)
            * Must: edges.csv (show all the edges)
            * Not Must: group-edges.csv (show the label of the nodes)
        * Write the config file in /config/${data_name}.ini
        * Train your model by `python train.sh`
        * Evaluate your model by `python evaluate.sh`
    * Demo
        * python train.sh
        * python evaluate.sh


### Model We Offered
    * AutoEncoder
    * Graph Factorization
    * Deep Walk
    * LINE
    * SDNE
    * GrapRep

### Others
    * Some scripts support very large network embedding, others not
    * Not all the model achieve the best as the origin papers say, like SDNE
    * Other good model will be coded sustainedly