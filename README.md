# LSM in Parlai

These files are part of the ParlAI framework. In order do use the code, they need to be implemented in the framework as follows:

Substitute the following files with this version:
```
  parlai/core/torch_generator_agent.py
  parlai/core/torch_agent.py
```

  
Add the lsm file to the same directory:
```
  parlai/core/lsm.py
```
  
In order to run the baseline and any of the objective models, the model can be run as follows:
```
  python -u ParlAI/examples/train_model.py \
    -m transformer/generator \
    -t dailydialog \
    -mf <model file to save the model in> \
    --inference beam \
    --lossdoc <file to document the loss during training>
```

In order to switch to the lsm objective that applies lsm on the golden standard during training time: \
  * In parlai/core/torch_generator_agent.py, uncomment line 703:
    ```
    #loss = LSM_loss_1(self, batch, loss)
    ```
  
In order to switch to the lsm objective that applies lsm during the sequence generation:
  * In parlai/core/torch_generator_agent.py, comment line 1037:
    ```
    b.advance(score[i])
    ```
  * In parlai/core/torch_generator_agent.py, uncomment line 1038:
    ```
    #b.my_advance(lsm_dict, score[i])
    ```
