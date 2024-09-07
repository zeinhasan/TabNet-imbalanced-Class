# TabNet-imbalanced-Class
TabNet : Attentive Interpretable Tabular Learning Case Imbalanced Class

# TabNet 
**TabNet** was introduced by researchers at Google Cloud AI in a 2019 paper titled "**TabNet: Attentive Interpretable Tabular Learning**." It was developed to address the unique challenges of applying deep learning to tabular data, which traditional deep learning models (like convolutional neural networks or recurrent neural networks) struggle with, despite their success in other domains like computer vision and natural language processing.

### Background and Motivation:

1. **Challenges with Tabular Data**:
   Tabular data—commonly seen in databases or spreadsheets—is structured into rows (samples) and columns (features), which can be a mix of categorical and numerical values. Traditional deep learning models are often designed for unstructured data like images or text, where relationships between features (e.g., pixels in an image) are spatial or sequential. However, tabular data typically lacks these kinds of relationships, making it harder for deep learning models to excel without overfitting.

2. **Dominance of Decision Trees**:
   Prior to TabNet, traditional machine learning methods like decision trees and their ensemble versions (e.g., Random Forest, Gradient Boosting Machines like XGBoost, LightGBM) dominated tabular data tasks. These models are naturally better suited to tabular data, as they can handle feature importance, missing values, and mixed feature types well.

3. **Need for Deep Learning with Tabular Data**:
   While tree-based methods work well, deep learning models have inherent advantages when dealing with large, high-dimensional data, and they can be integrated with other deep learning frameworks. Additionally, deep learning models like TabNet can learn end-to-end, support feature interactions in a flexible way, and leverage GPUs for faster training. Thus, TabNet aimed to bring the benefits of deep learning to tabular data, without losing the interpretability and feature selection strengths of traditional models.

4. **Attention Mechanism and Sparsity**:
   TabNet introduced an **attention mechanism** to selectively focus on the most relevant features at each decision step, mimicking how tree-based methods only use certain features for each split. The model also promotes **sparsity**, meaning that it doesn't use all features at every step, which is crucial for reducing overfitting and improving interpretability.

5. **End-to-End Learning**:
   TabNet is an **end-to-end differentiable model**, meaning it can be trained using gradient-based methods like other deep learning models. This allows for easy integration with existing deep learning frameworks and optimization techniques.

### Key Innovations:

- **Sequential Attention**: TabNet learns which features to use at each decision step through a sequential attention mechanism, allowing it to focus on the most informative parts of the data dynamically.
  
- **Sparse Feature Selection**: TabNet emphasizes sparse decision-making, which leads to both better performance and interpretability. Only a subset of features is used at each step, which reduces redundancy.

- **Interpretability**: Unlike many neural networks that are often considered "black boxes," TabNet is inherently interpretable. It can highlight which features were important for each decision, providing insights into the model's behavior and making it easier to trust.

# Architecture of TabNet
TabNet consists of multiple decision steps, and each decision step takes input from a shared feature representation. The input features are processed by a series of steps, where each step makes partial decisions and outputs new feature embeddings for subsequent steps. This architecture is designed to handle both feature selection and the decision-making process in parallel.
## A. Feature Transformer
The Feature Transformer is a deep neural network that transforms the input features into a new, more expressive space. It consists of several fully connected layers with batch normalization and non-linear activations (e.g., ReLU).

<p align="center">
  <img src="TabNet Featue Transformer.png" alt="Feature Transformer"/>
</p>

The feature transformer’s role is to learn high-level feature representations from the raw input data. It’s applied at the beginning of each decision step to extract meaningful information from the input data.
It is composed of shared and decision-specific layers. The shared layers are used across all decision steps, while the decision-specific layers are unique for each step, allowing the model to progressively refine its feature representation as more decision steps are applied. 

## B. Attentive Transformer
The Attentive Transformer determines which features should be attended to at each decision step. It acts as a feature selection mechanism by assigning an importance score to each feature using an attention mechanism.

<p align="center">
  <img src="TabNet attentive transformer.png" alt="Feature Transformer"/>
</p>


The attention mechanism learns to focus on the most important subset of features dynamically at each decision step.
The Attentive Transformer takes the feature embeddings from the previous step and computes a probability distribution over the features, selecting the most relevant features for the next decision step.

## C. TabNet Encoder
<p align="center">
  <img src="TabNet Encoder.png" alt="Feature Transformer"/>
</p>
The **TabNet Encoder** is a core part of the TabNet architecture responsible for transforming raw tabular data into a more meaningful representation for decision-making. It consists of two key components: the **Feature Transformer** and the **Attentive Transformer**. These work together in multiple decision steps to progressively refine and select features, eventually producing an output that can be used for classification or regression.

### Components of the TabNet Encoder:

#### 1. **Feature Transformer**:
The **Feature Transformer** is a multi-layer neural network designed to learn rich representations of the input data. It transforms the input features into a new feature space where meaningful patterns can be extracted. The transformer consists of both shared and step-specific layers:

- **Shared Layers**: These layers are applied at every decision step. The weights of these layers are shared across all decision steps to extract common feature representations.
- **Decision-Specific Layers**: These layers are unique to each decision step, allowing the model to refine features specifically for that step. This step-by-step refinement allows for more focused and specialized learning as the model progresses through the decision steps.

The output of the feature transformer is passed to two places:
- The **Attentive Transformer** for feature selection.
- The final decision-making process.

#### 2. **Attentive Transformer**:
The **Attentive Transformer** is responsible for selecting which features the model should focus on at each decision step. It does this by calculating an attention score for each feature. The higher the score, the more important the feature is for the current decision step. The attention mechanism works as follows:

- It takes as input the features produced by the Feature Transformer from the previous decision step.
- It outputs a probability distribution over the features, indicating the relevance of each feature.
  
Instead of using softmax (which produces dense probabilities), **Sparsemax** is used. Sparsemax encourages many feature attention scores to be zero, meaning that only a few features are selected at each step. This promotes **sparsity** and improves the interpretability of the model.

#### 3. **Decision Steps**:
The TabNet encoder operates through a sequence of **decision steps**, where each step focuses on different features selected by the Attentive Transformer. The number of decision steps is a hyperparameter of the model. Each decision step performs the following tasks:
- It uses the **Attentive Transformer** to select a subset of features.
- It passes these selected features through the **Feature Transformer** to extract information that contributes to the decision-making process.
- It produces a partial decision that contributes to the final prediction.

This process is repeated for multiple decision steps, allowing the model to progressively refine the feature selection and transformation.

#### 4. **Residual Connections**:
To prevent the loss of information, **residual connections** are employed in the encoder, similar to ResNet-style architectures. These connections allow the model to combine information from earlier decision steps with the output of the current step, leading to better gradient flow and more stable training.

---

### Flow of Data through the Encoder:

1. **Input Features**: Raw features are fed into the encoder.
2. **Feature Transformer**: The features are passed through the shared and decision-specific layers of the Feature Transformer, transforming them into a more meaningful representation.
3. **Attentive Transformer**: The transformed features are then passed to the Attentive Transformer, which selects the most relevant features for that particular decision step.
4. **Decision Steps**:
   - At each decision step, the selected features are refined by the Feature Transformer.
   - The partial decision for the current step is made and passed forward to the next step.
5. **Output**: After all decision steps are complete, the final output is obtained by aggregating the partial decisions from all steps.

---


## D. TabNet Decoder
<p align="center">
  <img src="TabNet Decoder.png" alt="Feature Transformer"/>
</p>
The **TabNet Decoder** is used in the context of **self-supervised learning** tasks like pretraining. It is paired with the TabNet Encoder to allow the model to reconstruct input data, providing a training signal for unsupervised feature learning. In this scenario, the model first learns to encode and then decode the data, with the goal of improving the encoder's ability to generate meaningful feature representations. Decoder. 
<br/>
<br/>
The final output of the decoder is a reconstructed version of the input features, which is compared against the original features to compute the reconstruction loss. This loss is used to update both the encoder and decoder during training.
<br/><br/>
The TabNet Decoder is typically used in a **self-supervised pretraining phase**. Here’s how it fits into the workflow:

1.  **Unsupervised Pretraining**: The
encoder-decoder pair is trained to reconstruct the input data. The encoder compresses the input into a latent representation, and the decoder attempts to reconstruct the original input from this representation. The reconstruction task forces the encoder to learn meaningful representations of the data, even without labels.
   
2. **Fine-Tuning for Supervised Learning**: After pretraining, the decoder is discarded, and the pretrained encoder is fine-tuned for the supervised task (such as classification or regression). The pretrained encoder already has a good understanding of the structure of the data, leading to faster and better convergence in the supervised task.

### Key Components of the TabNet Decoder:

#### 1. **Feature Reconstruction**:
   The TabNet Decoder mirrors the structure of the TabNet Encoder but performs the inverse operation. Instead of learning to transform features for prediction (as in the encoder), the decoder is responsible for reconstructing the original input features from the learned representations. This reconstruction task forces the model to learn high-quality latent feature representations in the encoder.

#### 2. **Decoder Layers**:
   The decoder consists of layers similar to the feature transformer layers in the encoder. These layers attempt to reconstruct the original input features by learning to invert the feature transformation process that was applied by the encoder.

   - **Decision-Specific Layers**: Just like the encoder, the decoder uses decision-specific layers to progressively reconstruct the input data at each step.
   - **Shared Layers**: These layers are shared across all decoding steps and serve as the backbone for feature reconstruction.

#### 3. **Reconstruction Loss**:
During the reconstruction process, a **reconstruction loss** (such as Mean Squared Error) is calculated between the input data and the output of the decoder. The goal is to minimize this reconstruction loss, which encourages the encoder to learn meaningful, compressible representations of the input data.

#### 4. **Inverse Attention Mechanism**:
   The decoder applies an inverse of the attention mechanism used in the encoder. Instead of selecting which features to focus on for decision-making, the attention mechanism is applied in reverse to recover the input features. This helps the decoder focus on reconstructing the most important features first, while learning to rebuild the less important features as it progresses through the decision steps.

#### 5. **Multiple Decision Steps**:
   Similar to the encoder, the decoder operates through a series of decision steps. In each step, a subset of the latent representations from the encoder is used to reconstruct the corresponding part of the original input. The decoder gradually refines the reconstruction as more decision steps are applied.

---

## E. Sparsemax Activation:
TabNet uses Sparsemax (instead of softmax) to enforce sparsity in the attention mechanism. Sparsemax encourages many feature attention weights to be exactly zero, meaning the model only attends to a few features at each step. This makes the model both more interpretable and efficient.

Sparsemax outputs sparse probabilities, meaning that some of the features will receive a weight of zero, and the remaining features will be emphasized. This encourages the model to focus on a smaller subset of features at each decision step.

## F. Decision Steps:
TabNet operates through multiple decision steps. Each decision step is a combination of feature selection (via the Attentive Transformer) and transformation (via the Feature Transformer). The model progressively refines its understanding of the data as it goes through each step.

At each step, part of the feature representation is used to make a decision (which contributes to the final prediction), and the rest of the features are passed forward to the next step for further refinement.
This process allows TabNet to gradually build its decision, focusing on different features at different steps.

# Visualization of TabNet Architecture:
Here's a high-level flow of the TabNet architecture:

1. Input Features → Feature Transformer
2. Feature Transformer → Outputs transformed features for attention and decision-making.
3. Attentive Transformer → Selects which features to attend to, using Sparsemax to enforce sparsity.
4. Decision Steps (multiple):
- Selected features are used for partial decision-making.
- Residual connections are used to retain information across steps.
- Feature Transformer refines the features for the next step.
5. Final Aggregation of the decisions across all steps for the final output.