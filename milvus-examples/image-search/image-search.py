"""
Image Search with Milvus

This script demonstrates how to use Milvus to search for similar images in a dataset.
It uses a subset of the ImageNet dataset and a ResNet-34 model for feature extraction.
"""

import os
import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pymilvus import MilvusClient

# Constants
MILVUS_URI = "http://app-milvus-xxxxxxxx.namespace.svc.cluster.local:19530"
COLLECTION_NAME = "image_embeddings"

EMBEDDING_MODEL = "resnet34"


class FeatureExtractor:
    """Feature extractor using ResNet-34 model from timm."""
    
    def __init__(self, modelname):
        """
        Initialize the feature extractor with a pretrained model.
        
        Args:
            modelname: Name of the model to use for feature extraction
        """
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()
        self.input_size = self.model.default_cfg["input_size"]
        config = resolve_data_config({}, model=modelname)
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        """
        Extract features from an image.
        
        Args:
            imagepath: Path to the image file
            
        Returns:
            Normalized feature vector
        """
        input_image = Image.open(imagepath).convert("RGB")
        input_image = self.preprocess(input_image)
        input_tensor = input_image.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
        feature_vector = output.squeeze().numpy()
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()


def setup_milvus_collection():
    """
    Set up Milvus collection for storing image embeddings.
    
    Returns:
        MilvusClient instance
    """
    client = MilvusClient(uri=MILVUS_URI)
    
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vector_field_name="vector",
        dimension=512,
        auto_id=True,
        enable_dynamic_field=True,
        metric_type="COSINE",
    )
    return client


def insert_embeddings(client, extractor, root_path):
    """
    Insert image embeddings into Milvus collection.
    
    Args:
        client: MilvusClient instance
        extractor: FeatureExtractor instance
        root_path: Root directory containing images
    """
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".JPEG"):
                filepath = os.path.join(dirpath, filename)
                try:
                    embedding = extractor(filepath)
                    client.insert(
                        COLLECTION_NAME,
                        {"vector": embedding, "filename": filepath}
                    )
                except Exception as e:
                    print(f"Error processing {filepath}: {str(e)}")


def main():
    """Main function to execute the image search workflow."""
    # Initialize Milvus collection
    client = setup_milvus_collection()
    
    extractor = FeatureExtractor(EMBEDDING_MODEL)

    # Insert training data embeddings
    insert_embeddings(client, extractor, "./train")
    
    # Perform sample search
    query_image = "./test/Afghan_hound/n02088094_4261.JPEG"
    
    # Execute search
    results = client.search(
        COLLECTION_NAME,
        data=[extractor(query_image)],
        output_fields=["filename"],
        search_params={"metric_type": "COSINE"},
    )
    
    # Display results (simplified for script environment)
    print("\nQuery Image:", query_image)
    print("\nTop 10 Results:")
    for idx, hit in enumerate(results[0][:10]):
        print(f"{idx+1}. {hit['entity']['filename']}")


if __name__ == "__main__":
    main()
