# Marsan at semeval-2022 task 6: isarcasm detection via t5 and sequence learners

The paper describes SemEval-2022’s shared task “Intended Sarcasm Detection in English and Arabic.” This task includes English and Arabic tweets with sarcasm and non-sarcasm samples and irony speech labels. The first two subtasks predict whether a text is sarcastic and the ironic category the sarcasm sample belongs to. The third one is to find the sarcastic sample from its non-sarcastic paraphrase. Deep neural networks have recently achieved highly competitive performance in many tasks. Combining deep learning with language models has also resulted in acceptable accuracy. Inspired by this, we propose a novel deep learning model on top of language models. On top of T5, this architecture uses an encoder module of the transformer, followed by LSTM and attention to utilizing past and future information, concentrating on informative tokens. Due to the success of the proposed model, we used the same architecture with a few modifications to the output layer in all three subtasks.


<p align="center" width="100%" height="20">
    <img width="33%" src="https://user-images.githubusercontent.com/86873813/220597764-0aeb72e0-682c-4b2c-b7a5-a9e18b6e3e65.png">
</p>


the paper for this implementation can be found [here](https://aclanthology.org/2022.semeval-1.137/)



# Usages
#### Train Model:

###### train task a:
> trainer_task_a_sarcasm_detection.py

###### train task b:
> trainer_task_b_multi_label.py

###### train task c:
> trainer_task_c_reparaphrase_detection.py


#### Test Model

###### inference task a:
> python inferencer_task_a_sarcasm_detection.py

###### inference task b,c:
> python inferencer_task_b_multi_label.py

###### inference ensemble models:
> inferencer_ensemble.py
