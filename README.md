# ashby-hackathon
Team Members:  
William Eustis (CompE Junior)

Aniruddha Mukherjee (CS + Stats Senior)

Richwell Perez (CS Senior)

Philip Chmielowiec (CompE Junior)


## Approach

### Not Discussed: 4D Conv UNet

This was our first approach- we abadoned it because of OOM difficulties.

**Related Files:**
- train.py
- models/convNd.py


### Encoder/Decoder
We first pretrain a 3D Conv model with based off of the MobileNetV2 inverted residual blocks in a U-Net like archiecture. 

The first half of this model, from the input frame to the bottleneck, will act as an encoder for the 3D cubes into some vector representation. The second half of the network will act a a decoder, converting vectors back into volumes. 
We choose to use the MobileNetV2 blocks because of their efficiency and low-parameter usage. 

**Related Files:**
- models/coatnet.py
- train_autoencoder.py


### Transformer
We would like to use context from time, but we don't want our model to become prohibitively expensive to run. Using a transformer allows us to input a dynamic number of timesteps. If you'd like more context for some prediction, you can just provide it with more encoded samples.

We use a standard transformer network (encoder only, GELU, full-attention) to accept some number of embeddings from the encoder (each embedding is representative of one time-step). This attention network processes the information between the diferent encodings as it prepares it for the decoder. 
This transformer will output as many sequence elements as are input, but we use only the first 10. Each output token will be used to generate some volume for a specific feature. 

**Related Files:**
-models/transf.py
-train large.py

### Misc
By training the encoder beforehand, we can freeze the weights while training the transformer and decoder, this allows us to prevent OOM errors.
One significant benefit of our approach is that you can change the number of timesteps you process at runtime. Though,  at least 10  timesteps must be used with the current architecture (but if we use one token to generate the full 10-channel volume, you can train on any number of timesteps).
We train the encoder/decoder with very high regularization, in the hopes that it will generalize well. We find that this improved results.


## Important Files
- train_autoencoder, train_large.py are the primary training scripts
- dataset, dataset_eval are the dataset files
- models/{coatnet, transf} are the primary model files
- shapy.py work on SHAP values for feature importance/selection

## Results

CHI :4.64%


CHI_CCN :6.13%


D_ALPHA :21.75%


D_ALPHA_CCN :6.11%


D_GAMMA :29.43%


D_GAMMA_CCN :7.90%


PM25 :155.56%


ccn_001 :51.95%


ccn_003 :63.43%


ccn_006 :60.34%

## Slides

https://docs.google.com/presentation/d/1Q9zXc-R9FRtJKiek2e3E4Fuqq1qkitqE/edit?usp=sharing&ouid=109565201557138189351&rtpof=true&sd=true


