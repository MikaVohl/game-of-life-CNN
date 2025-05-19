FROM public.ecr.aws/lambda/python:3.11
WORKDIR /var/task

COPY infra/lambda/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy only what Lambda needs: the package, handler, model
COPY src/life_sim ./life_sim
COPY infra/lambda/handler.py infra/lambda/life.onnx ./

CMD ["handler.handler"]