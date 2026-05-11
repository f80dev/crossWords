from os import environ
import os # Added for path manipulation

import anthropic
from  ollama import Client
import pymupdf
import requests
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import json
import re
import random
import io
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware


PROMPT= """ 
        Pour la liste de mots suivante : {words}, rédige une définition courte.
        Le niveau de vocabulaire doit être strictement adapté à un enfant de {age} ans.
        """

PERSONA="""
        Tu es un expert en création de mots croisés.
        Tu donnes des définitions courtes, en une seule phrase ou des synonymes adapté à l'age de ton lecteur. Tu ne cites jamais le mot à définir dans la définition
        Ne donne jamais d'exemple utilisant le mot.
        """
MODEL="google/gemma-4-26B-A4B-it"

app = FastAPI(title="Générateur de Mots Croisés API", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


class DocumentExtractor:
    @staticmethod
    def extract_from_pdf(file_bytes: bytes) -> str:
        text = ""
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            raise Exception(f"Erreur de lecture PDF: {str(e)}")

    @staticmethod
    def extract_from_epub(file_bytes: bytes) -> str:
        text = ""
        try:
            book = epub.read_epub(io.BytesIO(file_bytes))
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                text += soup.get_text(separator=' ')
            return text
        except Exception as e:
            raise Exception(f"Erreur de lecture EPUB: {str(e)}")


class CrosswordGenerator:
    def __init__(self, size_limit=30):
        self.grid = {}  # Utilisation d'un dictionnaire pour une grille "infinites"
        self.placed_words_info = []
        self.size_limit = size_limit
        self.dictionnaire={}
        self.load_dictionnaire() # Load dictionary on initialization


    def load_dictionnaire(self):
        """
        Charge le dictionnaire français depuis un fichier JSON.
        """
        script_dir = os.path.dirname(__file__) # Get the directory of the current script
        file_path = os.path.join(script_dir, "dictionnaire_francais.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.dictionnaire = json.load(f)
            print(f"Dictionnaire chargé avec {len(self.dictionnaire)} mots.")
        except FileNotFoundError:
            print(f"Erreur: Le fichier dictionnaire_francais.json n'a pas été trouvé à {file_path}")
            self.dictionnaire = {}
        except json.JSONDecodeError:
            print(f"Erreur: Impossible de décoder le fichier JSON à {file_path}")
            self.dictionnaire = {}
        except Exception as e:
            print(f"Une erreur inattendue est survenue lors du chargement du dictionnaire: {e}")
            self.dictionnaire = {}


    def extract_words(self, text: str,word_limit=1000000) -> List[str]:
        """Extrait les mots uniques, les mélange et les trie par longueur."""
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,15}\b', text.upper())
        unique_words = list(set(words))
        random.shuffle(unique_words)  # Mélange pour la variété
        return unique_words[:word_limit]

    def consulter_ddf(self,word):
        """
        Récupère l'introduction (souvent la définition principale) d'un mot.
        """
        HEADERS = {"User-Agent": "CrosswordGenBot/1.0 (hhoareau@gmail.com)"}
        params = {
            "action": "query",
            "format": "json",
            "titles": word,
            "prop": "extracts",
            "exintro": True,      # Récupère uniquement l'introduction
            "explaintext": True,  # Format texte brut au lieu de HTML
            "redirects": 1        # Suit les redirections (ex: verbe conjugué -> infinitif)
        }

        response = requests.get("https://fr.wiktionary.org/w/api.php", params=params,headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        # Extraction de la page dans le JSON (la clé est l'ID de la page)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None

        page_id = next(iter(pages))
        page_data = pages[page_id]

        if "missing" in page_data:
            return None

        return page_data.get("extract", "Aucune définition disponible.")



    def _can_place_word(self, word, row, col, direction):
        """Vérifie si un mot peut être placé sans collisions ou adjacences parallèles."""
        if row < 0 or col < 0:
            return False

        if row + len(word) > self.size_limit or col + len(word) > self.size_limit:
            return False

        if direction == "HORIZONTAL":
            # Vérifier les collisions et les voisins parallèles
            for i, letter in enumerate(word):
                current_pos = (row, col + i)
                # Collision : une lettre différente est déjà là
                if self.grid.get(current_pos) and self.grid[current_pos] != letter:
                    return False
                # Voisin parallèle (au-dessus ou en-dessous)
                if self.grid.get((row - 1, col + i)) or self.grid.get((row + 1, col + i)):
                    # Sauf si c'est le point d'intersection lui-même
                    if self.grid.get(current_pos) != letter:
                        return False
            # Vérifier les extrémités : pas de lettre juste avant ou juste après
            if self.grid.get((row, col - 1)) or self.grid.get((row, col + len(word))):
                return False
        else:  # VERTICAL
            for i, letter in enumerate(word):
                current_pos = (row + i, col)
                if self.grid.get(current_pos) and self.grid[current_pos] != letter:
                    return False
                if self.grid.get((row + i, col - 1)) or self.grid.get((row + i, col + 1)):
                    if self.grid.get(current_pos) != letter:
                        return False
            if self.grid.get((row - 1, col)) or self.grid.get((row + len(word), col)):
                return False



        return True

    def _calculate_score(self, word, row, col, direction):
        """Calcule le score d'un placement basé sur le nombre d'intersections."""
        score = 0
        if direction == "HORIZONTAL":
            for i, letter in enumerate(word):
                if self.grid.get((row, col + i)) == letter:
                    score += 1
        else:  # VERTICAL
            for i, letter in enumerate(word):
                if self.grid.get((row + i, col)) == letter:
                    score += 1
        return score

    def _place_word(self, word, row, col, direction):
        """Place un mot sur la grille et stocke ses informations."""
        if direction == "HORIZONTAL":
            for i, letter in enumerate(word):
                self.grid[(row, col + i)] = letter
        else:  # VERTICAL
            for i, letter in enumerate(word):
                self.grid[(row + i, col)] = letter

        self.placed_words_info.append({"word": word, "row": row, "col": col, "direction": direction})

    def _finalize_grid(self):
        """Convertit la grille de dictionnaire en une grille 2D et normalise les coordonnées."""
        if not self.grid:
            return [], []

        min_row = min(r for r, c in self.grid.keys())
        max_row = max(r for r, c in self.grid.keys())
        min_col = min(c for r, c in self.grid.keys())
        max_col = max(c for r, c in self.grid.keys())

        height = max_row - min_row + 1
        width = max_col - min_col + 1

        final_grid = [[' ' for _ in range(width)] for _ in range(height)]
        for (r, c), letter in self.grid.items():
            final_grid[r - min_row][c - min_col] = letter

        final_words_info = []
        for info in self.placed_words_info:
            final_words_info.append({
                "word": info["word"],
                "row": info["row"] - min_row,
                "col": info["col"] - min_col,
                "direction": info["direction"]
            })
        
        return final_grid, final_words_info

    def list_words(self):
        """
        Extrait tous les mots de la grille finalisée, horizontalement et verticalement.
        Utile pour la vérification ou si la liste de mots n'est pas disponible.
        """
        final_grid, _ = self._finalize_grid()
        if not final_grid:
            return []

        height = len(final_grid)
        width = len(final_grid[0])
        found_words = set()

        # Lecture horizontale
        for r in range(height):
            row_str = "".join(final_grid[r])
            # Trouve les séquences de lettres d'au moins 4 caractères
            words_in_row = re.findall(r'[A-ZÀ-ÿ]{4,}', row_str)
            for word in words_in_row:
                found_words.add(word)

        # Lecture verticale
        for c in range(width):
            col_str = "".join(final_grid[r][c] for r in range(height))
            # Trouve les séquences de lettres d'au moins 4 caractères
            words_in_col = re.findall(r'[A-ZÀ-ÿ]{4,}', col_str)
            for word in words_in_col:
                found_words.add(word)

        return list(found_words)

    def build_grid(self, text: str, words_limit=1000000,rest=10):
        """Construit la grille en utilisant l'heuristique de meilleur placement aléatoire."""
        words_to_place = self.extract_words(text,words_limit)

        # Place le premier mot (mot souche)
        seed_word = words_to_place.pop(0)
        self._place_word(seed_word,
                         random.randint(0, self.size_limit-len(seed_word)),
                         random.randint(0, self.size_limit-len(seed_word)) , "HORIZONTAL")

        # Boucle principale pour placer les mots restants
        defs=dict()

        for word in list(words_to_place):

            #recherche la définition
            if not word.lower() in self.dictionnaire.keys():
                self.dictionnaire[word] = self.consulter_ddf(word)

            if word.lower() in self.dictionnaire.keys():
                possible_placements = []
                # Chercher des intersections valides
                for i, letter_in_word in enumerate(word):
                    for (r, c), letter_on_grid in self.grid.items():
                        if letter_in_word == letter_on_grid:
                            # Tenter un placement vertical
                            if self._can_place_word(word, r - i, c, "VERTICAL"):
                                score = self._calculate_score(word, r - i, c, "VERTICAL")
                                possible_placements.append({"word": word, "row": r - i, "col": c, "direction": "VERTICAL", "score": score})
                            # Tenter un placement horizontal
                            if self._can_place_word(word, r, c - i, "HORIZONTAL"):
                                score = self._calculate_score(word, r, c - i, "HORIZONTAL")
                                possible_placements.append({"word": word, "row": r, "col": c - i, "direction": "HORIZONTAL", "score": score})

                if not possible_placements:
                    continue

                # Sélectionner parmi les meilleurs scores
                max_score = max(p['score'] for p in possible_placements)
                best_placements = [p for p in possible_placements if p['score'] == max_score]

                if best_placements:
                    chosen_placement = random.choice(best_placements)
                    self._place_word(chosen_placement["word"], chosen_placement["row"], chosen_placement["col"], chosen_placement["direction"])
                    words_to_place.remove(word)
                    if len(self.dictionnaire[word.lower()])>0 and "definition" in self.dictionnaire[word.lower()][0]:
                        defs[word]=self.dictionnaire[word.lower()][0]["definition"]
                        print(f"Placement de {word} définie par {self.dictionnaire[word.lower()]}. Il reste {rest} mots à placer")
                        rest=rest-1
                    if rest==0: break
                else:
                    pass

        final_grid, final_words_info = self._finalize_grid()

        return {
            "grid": final_grid,
            "placed_words": final_words_info,
            "words_list": [info["word"] for info in final_words_info],
            "definitions":defs
        }


class OllamaDefinitionProvider:
    def get_definitions(self, words: list, age: int) -> dict:
        client = Client(host='http://127.0.0.1:11434', timeout=30.0)
        response = client.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": PERSONA.format(age=age) + PROMPT.format(words=', '.join(words), age=age)
                }
            ]
        )
        try:
            content = response["message"]["content"]
            # Nettoyage pour extraire uniquement le JSON
            clean_text = content.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:-3].strip()
            elif clean_text.startswith('```'):
                clean_text = clean_text[3:-3].strip()
            return json.loads(clean_text)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Erreur de parsing JSON depuis Ollama: {e}")
            return {word: "Définition non valide" for word in words}



class MinimaxDefinitionProvider:
    def get_definitions(self, words: list[str], age: int,dictionnaire:dict) -> dict:


        client = anthropic.Anthropic(
            base_url="https://api.minimax.io/anthropic",
            api_key=environ.get("MINIMAX_API_KEY")  # Replace with your MiniMax API Key
        )

        rc=dict()



        response = client.messages.create(
            model="MiniMax-M2.7",
            max_tokens=500,
            system=[{"type": "text","text": PERSONA}],
            messages=[{"role": "user", "content": f"Donne une définition compréhensible pour un enfant de {age} ans et courte pour chacun des mots de la liste suivante : {','.join(words)}. Sépare chaque définition par un * en rappelant le mot définie au début"}]
        )


        if len(response.content)>1:
            reponse=response.content[1].text
            for definition in reponse.split(" — "):
                rc[definition.split(" - ")[0]]=definition.split(" - ")[1]

        return rc



extractor = DocumentExtractor()
definition_provider = MinimaxDefinitionProvider()

# ==========================================
# MODÈLES PYDANTIC (Validation des données)
# ==========================================

class WordInfo(BaseModel):
    word: str
    row: int
    col: int
    direction: str
    definition: str


class CrosswordExportRequest(BaseModel):
    grid: List[List[str]]
    words: List[WordInfo]



# ==========================================
# SERVICES
# ==========================================

#http://localhost:8000/api/v1/generate
@app.post("/api/v1/generate")
async def generate_crossword_endpoint(
        file: UploadFile = File(..., description="Fichier PDF ou EPUB"),
        age: int = Form(..., description="Âge cible pour les définitions"),
        grid_size: int = Form(20, description="Taille de la grille (ex: 20x20)")
):
    """
    Endpoint principal : Reçoit un fichier, extrait le texte, génère la grille
    et récupère les définitions adaptées à l'âge via Gemini.
    """
    # 1. Lecture du fichier binaire
    file_bytes = await file.read()
    filename = file.filename.lower()

    # 2. Extraction du texte
    if filename.endswith(".pdf"):
        text = extractor.extract_from_pdf(file_bytes)
    elif filename.endswith(".epub"):
        text = extractor.extract_from_epub(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Format non supporté. Utilisez PDF ou EPUB.")

    if not text:
        raise HTTPException(status_code=400, detail="Impossible d'extraire du texte de ce document.")

    # 3. Génération de la grille
    generator = CrosswordGenerator(size_limit=grid_size)
    grid_data = generator.build_grid(text,rest=int(grid_size))

    if not grid_data or not grid_data["placed_words"]:
        raise HTTPException(status_code=400, detail="Pas assez de mots valides pour générer une grille.")

    # 4. Récupération des définitions via Gemini
    #definitions = definition_provider.get_definitions(grid_data["words_list"], age,generator.dictionnaire)

    # 5. Construction de la réponse JSON finale pour le front-end ou n8n
    rc={
        "metadata": {
            "source_file": file.filename,
            "target_age": age,
            "grid_size": f"{len(grid_data['grid'][0])}x{len(grid_data['grid'])}"
        },
        "grid": grid_data["grid"],
        "words": grid_data["definitions"]
    }

    # for item in sorted(grid_data["placed_words"], key=lambda x: (x['row'], x['col'])):
    #     rc["words"].append({
    #         "word": item["word"],
    #         "row": item["row"],
    #         "col": item["col"],
    #         "direction": item["direction"],
    #         "definition": definitions[item["word"]]
    #     })

    return rc



# Lancement serveur de développement (à taper dans le terminal) :
# uvicorn main:app --reload

# ==========================================
# SERVICE DE GÉNÉRATION PDF
# ==========================================

class PDFExportService:
    @staticmethod
    def generate_pdf(data: CrosswordExportRequest) -> io.BytesIO:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # --- Page 1: Grille vide et Définitions ---

        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "Grille de Mots Croisés")

        cell_size = 20
        grid_width = len(data.grid[0]) * cell_size
        grid_height = len(data.grid) * cell_size
        x_offset = (width - grid_width) / 2
        y_offset = height - 100

        # Numérotation des colonnes (nombres)
        c.setFont("Helvetica", 10)
        for i in range(len(data.grid[0])):
            c.drawString(x_offset + i * cell_size + cell_size / 2 - 3, y_offset + 10, str(i + 1))

        # Numérotation des lignes (lettres)
        for i in range(len(data.grid)):
            c.drawString(x_offset - 15, y_offset - i * cell_size - cell_size / 2 + 3, chr(65 + i))

        # Dessin de la grille vide
        c.setLineWidth(1)
        for r_idx, row in enumerate(data.grid):
            for c_idx, cell in enumerate(row):
                x = x_offset + c_idx * cell_size
                y = y_offset - r_idx * cell_size - cell_size
                if cell != ' ':
                    c.setFillColor(colors.white)
                    c.rect(x, y, cell_size, cell_size, fill=1, stroke=1)
                else:
                    c.setFillColor(colors.black)
                    c.rect(x, y, cell_size, cell_size, fill=1, stroke=1)

        # Ajout des définitions
        y_text = y_offset - grid_height - 40
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_text, "Définitions :")
        y_text -= 25

        c.setFont("Helvetica", 10)
        for idx, word_info in enumerate(data.words):
            dir_fr = "Horiz." if word_info.direction == "HORIZONTAL" else "Vert."
            start_char = chr(65 + word_info.row)
            text_line = f"{idx + 1}. ({start_char}{word_info.col + 1} {dir_fr}) : {word_info.definition}"
            if y_text < 50:
                c.showPage()
                y_text = height - 50
                c.setFont("Helvetica", 10)
            c.drawString(50, y_text, text_line)
            y_text -= 15

        # --- Page 2: Solution ---
        c.showPage()
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "Solution de la Grille")

        # Redessiner la grille avec les réponses
        for r_idx, row in enumerate(data.grid):
            for c_idx, cell in enumerate(row):
                x = x_offset + c_idx * cell_size
                y = y_offset - r_idx * cell_size - cell_size
                if cell != ' ':
                    c.setFillColor(colors.white)
                    c.rect(x, y, cell_size, cell_size, fill=1, stroke=1)
                    c.setFillColor(colors.black)
                    c.setFont("Helvetica", 12)
                    c.drawCentredString(x + cell_size / 2, y + cell_size / 2 - 5, cell)
                else:
                    c.setFillColor(colors.black)
                    c.rect(x, y, cell_size, cell_size, fill=1, stroke=1)

        c.save()
        buffer.seek(0)
        return buffer


@app.post("/api/v1/export/pdf", response_class=StreamingResponse)
async def export_crossword_pdf(request_data: CrosswordExportRequest):
    """
    Reçoit les données JSON d'une grille et retourne un fichier PDF téléchargeable.
    """
    try:
        pdf_buffer = PDFExportService.generate_pdf(request_data)

        # Retourne le flux binaire avec les bons headers pour forcer le téléchargement
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=mots_croises.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du PDF: {str(e)}")



import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)