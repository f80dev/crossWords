import pytest
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Importation de notre application et de nos services depuis main.py
from main import app, CrosswordGenerator, GeminiDefinitionProvider, PDFExportService, CrosswordExportRequest

client = TestClient(app)


# ==========================================
# TESTS UNITAIRES : LOGIQUE MÉTIER
# ==========================================

def test_crossword_generator_extract_words():
    """Test l'extraction et le filtrage des mots."""
    generator = CrosswordGenerator(size=10)
    corpus = "Le développement de logiciels demande une architecture robuste."

    words = generator.extract_words(corpus)

    # Vérifie que les petits mots ("Le", "de", "une") sont ignorés (seuil > 4 ou 5 lettres selon ton implémentation)
    assert "DÉVELOPPEMENT" in words
    assert "ARCHITECTURE" in words
    assert "ROBUSTE" in words
    assert "LE" not in words


def test_crossword_generator_build_grid():
    """Test la construction basique de la grille."""
    generator = CrosswordGenerator(size=20)
    corpus = "Python"

    result = generator.build_grid(corpus)

    assert result is not None
    assert len(result["placed_words"]) == 1
    assert result["placed_words"][0]["word"] == "PYTHON"
    assert result["words_list"] == ["PYTHON"]


@patch('main.genai.GenerativeModel.generate_content')
def test_gemini_provider_mocked(mock_generate):
    """Test le fournisseur de définitions en simulant (mocking) l'API Gemini."""
    # Préparation du faux retour de l'API
    mock_response = MagicMock()
    mock_response.text = '{"PYTHON": "Un langage de programmation", "DOCKER": "Une baleine bleue"}'
    mock_generate.return_value = mock_response

    provider = GeminiDefinitionProvider()
    provider.api_key = "fake_key"  # Force une clé pour passer la condition

    definitions = provider.get_definitions(["PYTHON", "DOCKER"], 10)

    assert "PYTHON" in definitions
    assert definitions["PYTHON"] == "Un langage de programmation"
    mock_generate.assert_called_once()  # Vérifie que la méthode a bien été appelée


# ==========================================
# TESTS D'INTÉGRATION : ENDPOINTS FASTAPI
# ==========================================

@patch('main.DocumentExtractor.extract_from_pdf')
@patch('main.GeminiDefinitionProvider.get_definitions')
def test_generate_endpoint_with_pdf(mock_get_definitions, mock_extract_pdf):
    """Test l'endpoint de génération de grille avec un faux fichier PDF."""
    # 1. Simulation de l'extraction de texte
    mock_extract_pdf.return_value = "Python backend architecture"

    # 2. Simulation du retour de Gemini
    mock_get_definitions.return_value = {"PYTHON": "Langage cool"}

    # 3. Création d'un faux fichier en mémoire
    fake_file = io.BytesIO(b"dummy pdf content")

    # 4. Exécution de la requête POST via TestClient
    response = client.post(
        "/api/v1/generate",
        files={"file": ("test.pdf", fake_file, "application/pdf")},
        data={"age": 12, "grid_size": 30}
    )

    # 5. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["target_age"] == 12
    assert data["metadata"]["source_file"] == "test.pdf"
    assert len(data["words"]) > 0
    assert data["words"][0]["definition"] == "Langage cool"


@patch('main.GeminiDefinitionProvider.get_definitions')
def test_with_real_epub(mock_get_definitions):
    """Test l'endpoint avec un vrai fichier EPUB."""
    mock_get_definitions.return_value = {"TEST": "A test definition"}
    with open("test.epub", "rb") as f:
        response = client.post(
            "/api/v1/generate",
            files={"file": ("test.epub", f, "application/epub+zip")},
            data={"age": 8, "grid_size": 20}
        )
    assert response.status_code == 200
    data = response.json()

    with open("test.pdf", "wb") as f:
        f.write(PDFExportService().generate_pdf(CrosswordExportRequest(**data)).read())



@patch('main.DocumentExtractor.extract_from_epub')
@patch('main.GeminiDefinitionProvider.get_definitions')
def test_generate_endpoint_with_epub(mock_get_definitions, mock_extract_epub):
    """Test l'endpoint de génération de grille avec un faux fichier EPUB."""
    # 1. Simulation de l'extraction de texte depuis un EPUB
    mock_extract_epub.return_value = "EPUB est un format de livre numérique"

    # 2. Simulation du retour de Gemini
    mock_get_definitions.return_value = {"FORMAT": "Manière de présenter"}

    # 3. Création d'un faux fichier EPUB en mémoire
    # Le contenu binaire n'a pas d'importance car l'extraction est mockée
    fake_file = io.BytesIO(b"dummy epub content")

    # 4. Exécution de la requête POST
    response = client.post(
        "/api/v1/generate",
        files={"file": ("test.epub", fake_file, "application/epub+zip")},
        data={"age": 10, "grid_size": 10}
    )

    # 5. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["target_age"] == 10
    assert data["metadata"]["source_file"] == "test.epub"
    assert len(data["words"]) > 0
    # Le mot extrait doit être "FORMAT" car "livre" et "est" sont trop courts
    assert data["words"][0]["word"] == "FORMAT"
    assert data["words"][0]["definition"] == "Manière de présenter"

def test_export_pdf_endpoint():
    """Test l'endpoint d'export PDF avec un payload JSON valide."""
    # Payload correspondant au modèle Pydantic CrosswordExportRequest
    payload = {

        "grid": [
            ["P", "Y", "T", "H", "O", "N"],
            [" ", " ", " ", " ", " ", " "]
        ],
        "words": [
            {
                "word": "PYTHON",
                "row": 0,
                "col": 0,
                "direction": "HORIZONTAL",
                "definition": "Un super langage"
            }
        ]
    }

    response = client.post("/api/v1/export/pdf", json=payload)

    # Vérification que le fichier est bien généré et renvoyé
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment; filename=mots_croises.pdf" in response.headers["content-disposition"]
    # Vérifie que le contenu commence par la signature binaire d'un PDF (%PDF)
    assert response.content.startswith(b"%PDF")