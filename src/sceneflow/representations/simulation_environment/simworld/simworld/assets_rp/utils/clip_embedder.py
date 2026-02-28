"""This class wraps the HuggingFace CLIP model to do embeddings."""
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


class CLIPEmbedder:
    """This class Use the CLIP model to compute text and image embeddings for semantic similarity."""
    def __init__(self, model_ID='openai/clip-vit-large-patch14-336'):
        """Initialize the following attributes: model, processor, tokenizer, device.

        Args:
            model_ID: the name of the model to do the embeddings.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CLIPModel.from_pretrained(model_ID).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_ID)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_ID)

    def get_text_embedding(self, text: str):
        """Encode a text string into a feature vector using the CLIP model.

        Args:
            text: A string of natural language input.

        Returns:
            A NumPy array representing the embedding of the input text.
        """
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        features = self.model.get_text_features(**inputs)
        return features.cpu().detach().numpy()

    def get_image_embedding(self, image):
        """Encode an image into a feature vector using the CLIP model.

        Args:
            image: A PIL image or NumPy array input.

        Returns:
            A NumPy array representing the embedding of the input image.
        """
        image_input = self.processor(images=image, return_tensors='pt')['pixel_values'].to(self.device)
        features = self.model.get_image_features(image_input)
        return features.cpu().detach().numpy()
