import torch
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import your model components
from model.encoder_decoder import EncoderCNN, DecoderRNN
from model.vocabulary import Vocabulary

app = Flask(__name__)
CORS(app)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load checkpoint
checkpoint = torch.load('model/caption_model.pth', map_location=device)

vocab_builder = checkpoint['vocab']
vocab = vocab_builder.itos
vocab_size = len(vocab_builder)

# Recreate the model architectures
encoder = EncoderCNN().to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab_builder)).to(device)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),])


def generate_caption(encoder, decoder, image, idx2word, transform, device, max_len=20):
    
    image = transform(image).unsqueeze(0).to(device)
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        features = encoder(image)
        inputs = features.unsqueeze(1)
        states = None
        sampled_ids = []
        
        for _ in range(max_len):
            hiddens, states = decoder.lstm(inputs, states)
            outputs = decoder.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = decoder.embedding(predicted).unsqueeze(1)
    
    caption = []
    for word_id in sampled_ids:
        word = idx2word.get(word_id, "<unk>")
        if word == '<end>':
            break
        caption.append(word)
    
    return ' '.join(caption)


@app.route('/caption', methods=['POST'])
def handle_generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']

    try:
        image = Image.open(image_file).convert('RGB')
        caption = generate_caption(encoder, decoder, image, vocab, transform, device)
        return jsonify({'caption': caption}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == "__main__":
    app.run(debug=True)