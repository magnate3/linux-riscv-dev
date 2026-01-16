
+ server
```
# Could be any recent publish tag (I tested with 24.08), just use the same for all containers so that the TRT versions are the same  

# Export model into model repo 
docker run --gpus all -it --rm -v ${PWD}:/triton_example nvcr.io/nvidia/pytorch:24.08-py3 python /triton_example/export.py

# Start server 
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}:/triton_example nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/triton_example/model_repository
```

+ client 
```
# Get a sample image
wget -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"

# Query server
docker run -it --net=host -v ${PWD}:/triton_example nvcr.io/nvidia/tritonserver:24.08-py3-sdk bash -c "pip install torchvision && python /triton_example/client.py"
```

You should get an output like:
```
[b'12.460938:90' b'11.523438:92' b'9.656250:14' b'8.414062:136'
 b'8.210938:11']
```

Step 3: Building a Triton Client to Query the Server
----------------------------------------------------

Before proceeding, make sure to have a sample image on hand. If you don't
have one, download an example image to test inference. In this section, we 
will be going over a very basic client. For a variety of more fleshed out
examples, refer to the `Triton Client Repository <https://github.com/triton-inference-server/client/tree/main/src/python/examples>`__

::

   wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"

We then need to install dependencies for building a python client. These will 
change from client to client. For a full list of all languages supported by Triton,
please refer to `Triton's client repository <https://github.com/triton-inference-server/client>`__.

::

   pip install torchvision
   pip install attrdict
   pip install nvidia-pyindex
   pip install tritonclient[all]

Let's jump into the client. Firstly, we write a small preprocessing function to
resize and normalize the query image.

::

   import numpy as np
   from torchvision import transforms
   from PIL import Image
   import tritonclient.http as httpclient
   from tritonclient.utils import triton_to_np_dtype

   # preprocessing function
   def rn50_preprocess(img_path="img1.jpg"):
       img = Image.open(img_path)
       preprocess = transforms.Compose([
           transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ])
       return preprocess(img).numpy()

   transformed_img = rn50_preprocess()

Building a client requires three basic points. Firstly, we setup a connection
with the Triton Inference Server.

::

   # Setting up client
   client = httpclient.InferenceServerClient(url="localhost:8000")

Secondly, we specify the names of the input and output layer(s) of our model.

::

   inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
   inputs.set_data_from_numpy(transformed_img, binary_data=True)

   outputs = httpclient.InferRequestedOutput("output__0", binary_data=True, class_count=1000)

Lastly, we send an inference request to the Triton Inference Server.

::

   # Querying the server
   results = client.infer(model_name="resnet50", inputs=[inputs], outputs=[outputs])
   inference_output = results.as_numpy('output__0')
   print(inference_output[:5])

The output of the same should look like below:

::

   [b'12.468750:90' b'11.523438:92' b'9.664062:14' b'8.429688:136'
    b'8.234375:11']