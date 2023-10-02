import bentoml
from bentoml.io import Text
from bentoml.io import NumpyNdarray

runner = bentoml.mlflow.get("toxic-comment-classifier:latest").to_runner()
svc = bentoml.Service("toxic-comment-classifier", runners=[runner])


@svc.api(input=Text(), output=NumpyNdarray())
def predict(input_text: str):
    return runner.predict.run(input_text)
