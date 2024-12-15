# VisionZip Demo-Chat Code
Many people have found our demo interesting. Therefore, we provide the demo-chat code in this branch, hoping it will be helpful to you.

This code is based on the [LLaVA](https://github.com/haotian-liu/LLaVA) framework with modifications. To help with analysis, we've provided the original unmodified LLaVA code in the commit `49b53a00dfd3a2d62ab79b8fdf5bc20d41b17ab9`. You can compare this with the latest version to see the changes made.

## Demo Setup

To run the demo, follow the same steps as LLaVA:

1. In **Terminal 1**, start the controller:
   ```bash
   python -m llava.serve.controller --host 0.0.0.0 --port 10000
   ```

2. In **Terminal 2**, launch the Gradio web server:
   ```bash
   python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
   ```

3. In **Terminal 3**, start the model worker:
   ```bash
   python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-7b
   ```

