# AuthentiScan

AuthentiScan is a comprehensive AI content detection system that analyzes text, images, videos, and websites to identify AI-generated content. It uses advanced machine learning models and computer vision techniques to provide accurate detection results.

## Features

- Text Analysis: Detects AI-generated text using multiple models and metrics
- Image Analysis: Identifies AI-generated images through face consistency and artifact detection
- Video Analysis: Analyzes video content for AI generation patterns
- Website Analysis: Comprehensive analysis of entire websites
- RESTful API: Easy integration with other applications

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Maleleee/AuthentiScan.git
cd AuthentiScan
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Start the server:
```bash
uvicorn app.main:app --reload
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## API Endpoints

- `POST /api/analyze/text`: Analyze text content
- `POST /api/analyze/image`: Analyze image content
- `POST /api/analyze/video`: Analyze video content
- `POST /api/analyze/website`: Analyze website content

## Development

The project structure follows a modular design:
```
app/
├── core/           # Core configuration and utilities
├── models/         # Data models and schemas
├── services/       # Analysis services
└── main.py         # FastAPI application
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI API for text analysis
- Microsoft DeBERTa for text classification
- Facebook BART for zero-shot classification
- StyleGAN for image detection
- Various deepfake detection models 
