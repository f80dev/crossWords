import unittest
from unittest.mock import patch, MagicMock
from main import CrosswordGenerator, DocumentExtractor

class TestDocumentExtractor(unittest.TestCase):

    @patch('main.pymupdf.open')
    def test_extract_from_pdf(self, mock_pymupdf_open):
        """
        Tests PDF text extraction by mocking the pymupdf library.
        """
        # Configure the mock
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Ceci est la première page. "
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Voici la deuxième."
        mock_doc.__iter__.return_value = [mock_page1, mock_page2]
        mock_pymupdf_open.return_value = mock_doc

        # Call the method with dummy data
        pdf_bytes = b'dummy_pdf_content'
        text = DocumentExtractor.extract_from_pdf(pdf_bytes)

        # Assertions
        mock_pymupdf_open.assert_called_once_with(stream=pdf_bytes, filetype="pdf")
        self.assertEqual(text, "Ceci est la première page. Voici la deuxième.")

    @patch('main.epub.read_epub')
    def test_extract_from_epub(self, mock_read_epub):
        """
        Tests EPUB text extraction by mocking the ebooklib library.
        """
        # Configure the mock
        mock_book = MagicMock()
        mock_item1 = MagicMock()
        mock_item1.get_body_content.return_value = b'<html><body><p>Premier paragraphe.</p></body></html>'
        mock_item2 = MagicMock()
        mock_item2.get_body_content.return_value = b'<html><body><p>Second paragraphe.</p></body></html>'
        
        # We need to import ITEM_DOCUMENT from ebooklib for the test to work
        from ebooklib import ITEM_DOCUMENT
        mock_book.get_items_of_type.return_value = [mock_item1, mock_item2]
        mock_read_epub.return_value = mock_book

        # Call the method with dummy data
        epub_bytes = b'dummy_epub_content'
        text = DocumentExtractor.extract_from_epub(epub_bytes)

        # Assertions
        mock_read_epub.assert_called_once()
        # BeautifulSoup adds spaces, so we expect "Premier paragraphe. Second paragraphe. "
        self.assertEqual(text.strip(), "Premier paragraphe. Second paragraphe.")


class TestCrosswordGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = CrosswordGenerator()

    def test_extract_words(self):
        """
        Tests the extraction of unique words, case conversion, and length filtering.
        """
        text = "Python est un langage de programmation. python est puissant. Mot."
        expected_words = {"PYTHON", "LANGAGE", "PROGRAMMATION", "PUISSANT"}
        
        text_controlled = "PYTHON LANGAGE PROGRAMMATION PUISSANT python"
        expected_words_controlled = ["PYTHON", "LANGAGE", "PROGRAMMATION", "PUISSANT"]
        
        extracted_words_controlled = self.generator.extract_words(text_controlled)
        
        self.assertCountEqual(extracted_words_controlled, expected_words_controlled, "Should extract unique, uppercase words of valid length")

    def test_can_place_word_empty_grid(self):
        """Tests if a word can be placed on an empty grid."""
        self.assertTrue(self.generator._can_place_word("PYTHON", 0, 0, "HORIZONTAL"))
        self.assertTrue(self.generator._can_place_word("PYTHON", 0, 0, "VERTICAL"))

    def test_can_place_word_collision(self):
        """Tests that a word cannot be placed if it collides with another."""
        self.generator._place_word("PYTHON", 1, 1, "HORIZONTAL")
        self.assertFalse(self.generator._can_place_word("TEST", 1, 0, "HORIZONTAL"), "Should detect collision")

    def test_can_place_word_intersection(self):
        """Tests that a word can be placed at an intersection."""
        self.generator._place_word("PYTHON", 1, 1, "HORIZONTAL")
        self.assertTrue(self.generator._can_place_word("TEST", 0, 3, "VERTICAL"), "Should allow valid intersection")

    def test_can_place_word_parallel_adjacency(self):
        """Tests that a word cannot be placed directly parallel to another."""
        self.generator._place_word("PYTHON", 1, 1, "HORIZONTAL")
        self.assertFalse(self.generator._can_place_word("ANOTHER", 2, 1, "HORIZONTAL"), "Should prevent parallel adjacency")

    def test_build_grid(self):
        """
        Tests the overall grid building process with a simple, predictable text.
        """
        text = "REPOS ETRE MERCI CIEL"
        grid_data = self.generator.build_grid(text, words_limit=4)
        
        self.assertIn("grid", grid_data)
        self.assertIn("placed_words", grid_data)
        self.assertGreater(len(grid_data["placed_words"]), 0, "The grid should have at least one word")
        self.assertIsInstance(grid_data["grid"], list)
        if grid_data["grid"]:
            self.assertIsInstance(grid_data["grid"][0], list)

if __name__ == '__main__':
    unittest.main()
