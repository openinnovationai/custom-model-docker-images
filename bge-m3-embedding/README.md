# BAAI/bge-m3 embedding model
A custom docker image to deploy BAAI/bge-m3 on MI210 GPU using Ray Serve


## to call the model infernece, the input should be on this format

```json
{
    "sentences": ["sentence 1", "sentence 2", "sentence 3"]
}
```

## image build
```sh
docker build -t bge-embedding-rocm .
```
