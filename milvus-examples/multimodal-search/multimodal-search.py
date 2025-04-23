"""Multimodal Retrieval with Amazon Reviews Dataset and LLVM Reranking.

This script implements a multimodal retrieval system that:
1. Loads and embeds product images using a visual embedding model
2. Stores embeddings in a Milvus vector database
3. Performs multimodal similarity search with image+text queries
4. Reranks results using a Large Language-Vision Model (LLVM)
"""

import os
import numpy as np
import cv2
from glob import glob
from typing import Dict, List, Tuple, Union
from PIL import Image
from tqdm import tqdm

import torch
from visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Constants

# In this example, the Milvus instance deployed using the official Helm chart has a bug,
# so a local instance is used temporarily.
# MILVUS_URI = "http://app-milvus-xxxxxxxx.namespace.svc.cluster.local:19530"
MILVUS_URI = "multimodal_demo.db"
COLLECTION_NAME = "amazon_reviews_2023"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL_PATH = "./Visualized_base_en_v1.5.pth"
EMBEDDING_DIM = 768
RERANK_MODEL = "Qwen2.5-VL-7B-Instruct"

DATA_DIR = "./images_folder"
IMG_HEIGHT = 300
IMG_WIDTH = 300
ROW_COUNT = 3


class LLVMGenerator:
    """Large Language-Vision Model generator for image captioning and reranking."""
    
    def __init__(self, model_id: str):
        """Initialize the LLVM generator with the specified model.
        
        Args:
            model_id: The ID of the model to use
        """
        # Load the model with recommended configurations
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",  # Automatically distribute across available devices
            trust_remote_code=True
        )
        
        # Load the processor with default settings
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    def generate(self, prompt: str, images: Union[str, List[str]]) -> str:
        """Generate text based on an image and prompt.
        
        Args:
            prompt: The text prompt to use
            images: Path to image or list of image paths
            
        Returns:
            Generated text response
        """
        # Handle single image path or multiple image paths
        if isinstance(images, str):
            image_list = [Image.open(images)]
        else:
            image_list = [Image.open(img_path) for img_path in images]
        
        # Prepare messages in the required format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in image_list
                ] + [{"type": "text", "text": prompt}]
            }
        ]
        
        # Apply chat template to format the input
        formatted_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision information (images and videos)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Tokenize and prepare inputs for the model
        inputs = self.processor(
            text=[formatted_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Define generation parameters
        generation_args = {
            "max_new_tokens": 500,  # Maximum number of tokens to generate
            "temperature": 0.0,     # Deterministic generation (no sampling)
            "do_sample": False      # Disable sampling
        }

        # Generate output
        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )

        # Trim input tokens and decode the generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text


class Encoder:
    """Encoder class for embedding images and text-image pairs."""
    
    def __init__(self, model_name: str, model_path: str):
        """Initialize the encoder with the specified model.
        
        Args:
            model_name: The name of the base model
            model_path: Path to the model weights
        """
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> List[float]:
        """Encode an image-text pair query.
        
        Args:
            image_path: Path to the query image
            text: Text instruction for the query
            
        Returns:
            List of embedding values
        """
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> List[float]:
        """Encode a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of embedding values
        """
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def setup_milvus(client: MilvusClient) -> None:
    """Create a collection in Milvus for storing embeddings.
    
    Args:
        client: Milvus client instance
    """
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    client.create_collection(COLLECTION_NAME, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", 
        index_type="IVF_FLAT",
        metric_type="COSINE"
    )
    
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )

    client.load_collection(
        collection_name=COLLECTION_NAME,
        replica_number=1
    )

    print(f"Collection '{COLLECTION_NAME}' created")


def generate_image_embeddings(encoder: Encoder, data_dir: str) -> Dict[str, List[float]]:
    """Generate embeddings for all images in the dataset.
    
    Args:
        encoder: Encoder instance for creating embeddings
        data_dir: Directory containing the images
        
    Returns:
        Dictionary mapping image paths to embeddings
    """
    image_list = glob(os.path.join(data_dir, "images", "*.jpg"))
    
    image_dict = {}
    for image_path in tqdm(image_list, desc="Generating image embeddings: "):
        try:
            image_dict[image_path] = encoder.encode_image(image_path)
        except Exception as e:
            print(f"Failed to generate embedding for {image_path}. Skipped.")
            continue
            
    return image_dict


def store_embeddings(client: MilvusClient, image_dict: Dict[str, List[float]]) -> None:
    """Store image embeddings in Milvus.
    
    Args:
        client: Milvus client instance
        image_dict: Dictionary of image paths and embeddings
    """
    data = []
    client.insert(
        collection_name=COLLECTION_NAME,
        data=[{"image_path": k, "vector": v} for k, v in image_dict.items()],
    )
    print(f"Inserted {len(image_dict)} embeddings into Milvus")


def search_similar_images(
    client: MilvusClient, 
    encoder: Encoder, 
    query_image: str, 
    query_text: str, 
    limit: int = 9
) -> List[str]:
    """Search for similar images using a multimodal query.
    
    Args:
        client: Milvus client instance
        encoder: Encoder instance
        query_image: Path to the query image
        query_text: Query text instruction
        limit: Maximum number of results to return
        
    Returns:
        List of paths to retrieved images
    """
    query_vec = encoder.encode_query(image_path=query_image, text=query_text)

    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        output_fields=["image_path"],
        limit=limit,
        search_params={"metric_type": "COSINE", "params": {}},
    )[0]

    return [hit.get("entity").get("image_path") for hit in search_results]


def create_panoramic_view(query_image_path: str, retrieved_images: List[str]) -> np.ndarray:
    """Create a panoramic view combining query and retrieved images.
    
    Args:
        query_image_path: Path to the query image
        retrieved_images: List of paths to retrieved images
        
    Returns:
        Numpy array containing the panoramic image
    """
    panoramic_width = IMG_WIDTH * ROW_COUNT
    panoramic_height = IMG_HEIGHT * ROW_COUNT
    panoramic_image = np.full(
        (panoramic_height, panoramic_width, 3), 255, dtype=np.uint8
    )

    # Create and resize the query image with a blue border
    query_image_null = np.full((panoramic_height, IMG_WIDTH, 3), 255, dtype=np.uint8)
    query_image = Image.open(query_image_path).convert("RGB")
    query_array = np.array(query_image)[:, :, ::-1]
    resized_image = cv2.resize(query_array, (IMG_WIDTH, IMG_HEIGHT))

    border_size = 10
    blue = (255, 0, 0)  # blue color in BGR
    bordered_query_image = cv2.copyMakeBorder(
        resized_image,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=blue,
    )

    query_image_null[IMG_HEIGHT * 2 : IMG_HEIGHT * 3, 0:IMG_WIDTH] = cv2.resize(
        bordered_query_image, (IMG_WIDTH, IMG_HEIGHT)
    )

    # Add text "query" below the query image
    text = "query"
    font_scale = 1
    font_thickness = 2
    text_org = (10, IMG_HEIGHT * 3 + 30)
    cv2.putText(
        query_image_null,
        text,
        text_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        blue,
        font_thickness,
        cv2.LINE_AA,
    )

    # Combine the rest of the images into the panoramic view
    retrieved_imgs = [
        np.array(Image.open(img).convert("RGB"))[:, :, ::-1] for img in retrieved_images
    ]
    for i, image in enumerate(retrieved_imgs):
        image = cv2.resize(image, (IMG_WIDTH - 4, IMG_HEIGHT - 4))
        row = i // ROW_COUNT
        col = i % ROW_COUNT
        start_row = row * IMG_HEIGHT
        start_col = col * IMG_WIDTH

        border_size = 2
        bordered_image = cv2.copyMakeBorder(
            image,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        panoramic_image[
            start_row : start_row + IMG_HEIGHT, start_col : start_col + IMG_WIDTH
        ] = bordered_image

        # Add red index numbers to each image
        text = str(i + 1)
        org = (start_col + 50, start_row + 30)
        (font_width, font_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        )

        top_left = (org[0] - 48, start_row + 2)
        bottom_right = (org[0] - 48 + font_width + 5, org[1] + baseline + 5)

        cv2.rectangle(
            panoramic_image, top_left, bottom_right, (255, 255, 255), cv2.FILLED
        )
        cv2.putText(
            panoramic_image,
            text,
            (start_col + 10, start_row + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Combine the query image with the panoramic view
    panoramic_image = np.hstack([query_image_null, panoramic_image])
    return panoramic_image


def caption_image(image_path: str, llvm_generator: LLVMGenerator) -> str:
    """Generate a caption for an image using the LLVM.
    
    Args:
        image_path: Path to the image
        llvm_generator: Instance of LLVMGenerator
        
    Returns:
        Generated caption text
    """
    prompt = "You are a helpful assistant that captions images with short, descriptive, informative text no longer than a sentence. Caption this image."
    caption = llvm_generator.generate(prompt=prompt, images=image_path)
    return caption.removesuffix("<|end|>")


def rerank_results(
    query_text: str, 
    query_caption: str, 
    combined_image_path: str,
    llvm_generator: LLVMGenerator
) -> str:
    """Rerank the search results using an LLVM.
    
    Args:
        query_text: The original query text
        query_caption: Caption of the query image
        combined_image_path: Path to the combined panoramic image
        llvm_generator: Instance of LLVMGenerator
        
    Returns:
        The LLVM's response with reranking
    """
    information = (
        "You are responsible for ranking results for a Composed Image Retrieval. "
        "The user retrieves an image with an 'instruction' indicating their retrieval intent. "
        "For example, if the user queries a red car with the instruction 'change this car to blue,' a similar type of car in blue would be ranked higher in the results. "
        "Now you would receive instruction and an image containing the query image and multiple result images. The query image has a blue border and relates to the text instruction, and the result images have a red index number in their top left. Do not misunderstand it!"
        f"User instruction: {query_text} \n\n"
        f"Caption of query image: {query_caption} \n\n"
        "Provide a new ranked list of indices from most suitable to least suitable, followed by an explanation for the top 1 most suitable item only."
        "The format of the response has to be 'Ranked list: []' with the indices in brackets as integers, followed by 'Reasons:' plus the explanation why this most fit user's query intent. Ranked list includes 3 elements."
    )
    
    return llvm_generator.generate(prompt=information, images=combined_image_path)


def main():
    """Main function to run the multimodal retrieval pipeline."""
    # Initialize encoder and clients
    encoder = Encoder(EMBEDDING_MODEL, EMBEDDING_MODEL_PATH)
    milvus_client = MilvusClient(uri=MILVUS_URI)
    
    # Initialize LLVM generator
    llvm_generator = LLVMGenerator(model_id="./" + RERANK_MODEL)
    
    # Setup Milvus and generate embeddings
    setup_milvus(milvus_client)

    image_dict = generate_image_embeddings(encoder, DATA_DIR)
    store_embeddings(milvus_client, image_dict)
    
    # Define query
    query_image = os.path.join(DATA_DIR, "leopard.jpg")
    query_text = "phone case with this image theme"
    
    # Search for similar images
    print(f"Searching for images similar to {query_image} with text: '{query_text}'")
    retrieved_images = search_similar_images(
        milvus_client, encoder, query_image, query_text
    )
    
    # Create visual representation of results
    combined_image_path = os.path.join(DATA_DIR, "combined_image.jpg")
    panoramic_image = create_panoramic_view(query_image, retrieved_images)
    cv2.imwrite(combined_image_path, panoramic_image)

    # Caption the query image
    query_caption = caption_image(query_image, llvm_generator)
    print(f"Query image caption: {query_caption}")

    # Rerank results with LLVM
    print("Reranking results with LLVM...")
    reranking_result = rerank_results(query_text, query_caption, combined_image_path, llvm_generator)
    print(f"Reranking result: {reranking_result}")
    
    # Process reranked results (hardcoded example)
    # In a real application, you would parse the LLVM output to get the reranked indices
    reranked_indices = [7, 1, 2]
    reranked_retrieved_images = [retrieved_images[i - 1] for i in reranked_indices]
    
    # Create final visualization with reranked results
    reranked_combined_image_path = os.path.join(DATA_DIR, "reranked_combined_image.jpg")
    reranked_panoramic_image = create_panoramic_view(query_image, reranked_retrieved_images)
    cv2.imwrite(reranked_combined_image_path, reranked_panoramic_image)
    
    print(f"Original results saved to: {combined_image_path}")
    print(f"Reranked results saved to: {reranked_combined_image_path}")


if __name__ == "__main__":
    main()
