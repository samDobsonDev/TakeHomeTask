# Extensibility Guide

This document explains how to extend the content moderation system with new models, prediction categories and modalities.

---

## 1. Adding a New Model and Prediction Category

Each model produces predictions for a specific category. To add a new model, you need to:
1. Add the new category to the `Category` enum
2. Create a prediction class that returns the new category
3. Create a model class that implements the prediction logic
4. Register the model in the service container

The category is determined by the `get_category()` method on the prediction class, which returns a `Category` enum value. The category automatically appears in moderation results.

### Files to Modify

- `src/model.py` - Add category enum value, prediction class, and model
- `src/request_handler.py` - Register the model

### Example: Adding a Spam Detection Model

#### Step 1: Add the Category Enum Value

In `src/model.py`, add to the `Category` enum:

```python
class Category(Enum):
    """Content moderation categories"""
    HATE_SPEECH = "hate_speech"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    SPAM = "spam"  # Add new category
```

#### Step 2: Create the Prediction Class

Add to `src/model.py`:

```python
@dataclass
class SpamPrediction(ModelPrediction):
    """Prediction for spam content detection"""
    spam_score: float = 0.0
    promotional: float = 0.0
    phishing: float = 0.0
    scam: float = 0.0

    @classmethod
    def get_category(cls) -> Category:
        return Category.SPAM  # Return the enum value
```

#### Step 3: Create the Model Class

Add to `src/model.py`:

```python
class RandomSpamModel(ContentModerationModel[SpamPrediction]):
    """Model that detects spam content"""

    @property
    def name(self) -> str:
        return "RandomSpamModel"

    async def predict_text(self, input_data: PreprocessedText) -> SpamPrediction:
        return SpamPrediction(
            input_data=input_data,
            model_name=self.name,
            spam_score=random.random(),
            promotional=random.random(),
            phishing=random.random(),
            scam=random.random()
        )

    async def predict_image(self, input_data: PreprocessedImage) -> SpamPrediction:
        return SpamPrediction(
            input_data=input_data,
            model_name=self.name,
            spam_score=random.random(),
            promotional=random.random(),
            phishing=random.random(),
            scam=random.random()
        )
```

#### Step 4: Register the Model

In `src/request_handler.py`, update the `ServiceContainer.models` property:

```python
@property
def models(self) -> list[ContentModerationModel]:
    """Lazy-load models"""
    if self._models is None:
        self._models = [
            RandomHateSpeechModel(),
            RandomSexualModel(),
            RandomViolenceModel(),
            RandomSpamModel(),  # Add the new model
        ]
    return self._models
```

The new "spam" category will automatically appear in moderation results. The service uses `category.value` (the string `"spam"`) as the dict key, so results can be accessed via `response.results[Category.SPAM.value]` or `response.results["spam"]`.

---

## 2. Adding a New Modality (AUDIO)

Adding a new modality requires changes across multiple files since it affects the entire processing pipeline.

### Files to Modify

- `src/service.py` - Add enum value and routing logic
- `src/preprocessor.py` - Add preprocessed content type and preprocessor
- `src/model.py` - Add prediction method to interface and implementations
- `src/request_handler.py` - Register preprocessor and add validation

### Step 1: Add the Modality Enum Value

In `src/service.py`:

```python
class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"  # Add new modality
```

### Step 2: Create Preprocessed Content Type and Preprocessor

In `src/preprocessor.py`:

```python
@dataclass
class PreprocessedAudio(PreprocessedContent):
    """Preprocessed audio content"""
    data: list[int]  # Audio features/embeddings
    original_bytes: bytes
    duration_seconds: float
    sample_rate: int


class AudioPreprocessor(ContentPreprocessor):
    """Preprocessor for audio content"""

    def preprocess(self, content: bytes) -> PreprocessedAudio:
        # Process audio bytes - extract features, compute duration, etc.
        return PreprocessedAudio(
            data=[hash(content) % 256 for _ in range(128)],
            original_bytes=content,
            duration_seconds=0.0,  # Would be computed from actual audio
            sample_rate=44100
        )
```

### Step 3: Add Prediction Method to Model Interface

In `src/model.py`, update `ContentModerationModel`:

```python
class ContentModerationModel(ABC, Generic[PredictionType]):
    """Abstract base class for content moderation models"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def predict_text(self, input_data: PreprocessedText) -> PredictionType:
        pass

    @abstractmethod
    async def predict_image(self, input_data: PreprocessedImage) -> PredictionType:
        pass

    async def predict_video(self, input_data: PreprocessedVideo) -> list[PredictionType]:
        """Default: process each frame as an image"""
        return [await self.predict_image(frame) for frame in input_data.frames]

    @abstractmethod
    async def predict_audio(self, input_data: PreprocessedAudio) -> PredictionType:
        """Run prediction on audio content"""
        pass
```

### Step 4: Implement in All Model Classes

Update each model in `src/model.py`:

```python
class RandomHateSpeechModel(ContentModerationModel[HateSpeechPrediction]):
    # ... existing methods ...

    async def predict_audio(self, input_data: PreprocessedAudio) -> HateSpeechPrediction:
        return HateSpeechPrediction(
            input_data=input_data,
            model_name=self.name,
            toxicity=random.random(),
            severe_toxicity=random.random(),
            obscene=random.random(),
            insult=random.random(),
            identity_attack=random.random(),
            threat=random.random()
        )


class RandomSexualModel(ContentModerationModel[SexualPrediction]):
    # ... existing methods ...

    async def predict_audio(self, input_data: PreprocessedAudio) -> SexualPrediction:
        return SexualPrediction(
            input_data=input_data,
            model_name=self.name,
            sexual_explicit=random.random(),
            adult_content=random.random(),
            adult_toys=random.random()
        )


class RandomViolenceModel(ContentModerationModel[ViolencePrediction]):
    # ... existing methods ...

    async def predict_audio(self, input_data: PreprocessedAudio) -> ViolencePrediction:
        return ViolencePrediction(
            input_data=input_data,
            model_name=self.name,
            violence=random.random(),
            firearm=random.random(),
            knife=random.random()
        )
```

### Step 5: Update Content Decoding

In `src/service.py`, update `_decode_content`:

```python
def _decode_content(self, content: str | list[str], modality: Modality) -> str | bytes | list[bytes]:
    """Decode content based on modality"""
    if modality == Modality.TEXT:
        return content
    elif modality in (Modality.IMAGE, Modality.AUDIO):
        return base64.b64decode(content)
    elif modality == Modality.VIDEO:
        return [base64.b64decode(frame) for frame in content]
    else:
        raise ValueError(f"Unsupported modality: {modality}")
```

### Step 6: Update Prediction Routing

In `src/service.py`, update `_predict_by_modality`:

```python
async def _predict_by_modality(
    self,
    model: ContentModerationModel,
    preprocessed: PreprocessedContent,
    modality: Modality
) -> ModelPrediction | list[ModelPrediction]:
    """Route prediction to appropriate model method based on modality"""
    if modality == Modality.TEXT:
        return await model.predict_text(preprocessed)
    elif modality == Modality.IMAGE:
        return await model.predict_image(preprocessed)
    elif modality == Modality.VIDEO:
        return await model.predict_video(preprocessed)
    elif modality == Modality.AUDIO:
        return await model.predict_audio(preprocessed)
    else:
        raise ValueError(f"Unsupported modality: {modality}")
```

### Step 7: Register the Preprocessor

In `src/request_handler.py`, update `ServiceContainer.preprocessors`:

```python
@property
def preprocessors(self) -> dict[str, ContentPreprocessor]:
    """Lazy-load preprocessors"""
    if self._preprocessors is None:
        self._preprocessors = {
            "text": TextPreprocessor(),
            "image": ImagePreprocessor(),
            "video": VideoPreprocessor(),
            "audio": AudioPreprocessor(),  # Add audio preprocessor
        }
    return self._preprocessors
```

### Step 8: Add Request Validation

In `src/request_handler.py`, update `parse_request`:

```python
def parse_request(request_data: dict) -> ModerationRequest:
    # ... existing validation ...

    if modality == Modality.VIDEO:
        if not isinstance(content, list):
            raise ValueError("Video content must be a list of base64-encoded frames")
        if len(content) == 0:
            raise ValueError("Video content cannot be empty")
        if not all(isinstance(frame, str) for frame in content):
            raise ValueError("All video frames must be base64-encoded strings")
    elif modality == Modality.AUDIO:  # Add audio validation
        if not isinstance(content, str):
            raise ValueError("Audio content must be a base64-encoded string")
        if not content:
            raise ValueError("Content cannot be empty")
    else:
        if not isinstance(content, str):
            raise ValueError(f"{modality.value.capitalize()} content must be a base64-encoded string")
        if not content:
            raise ValueError("Content cannot be empty")

    # ... rest of function ...
```

---

## Summary

| Scenario               | Files to Modify                                                                   |
|------------------------|-----------------------------------------------------------------------------------|
| **New Model/Category** | `src/model.py`, `src/request_handler.py`                                          |
| **New Modality**       | `src/service.py`, `src/preprocessor.py`, `src/model.py`, `src/request_handler.py` |

The architecture follows these design principles:
- **Strategy Pattern**: Models implement a common interface
- **Factory Pattern**: Preprocessors are created based on modality
- **Generic Types**: `ContentModerationModel[PredictionType]` ensures type safety
- **Dependency Injection**: `ServiceContainer` manages all dependencies
