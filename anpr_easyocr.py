import easyocr
import cv2
import os
import re
import itertools
import unicodedata

class ArgentinePlateRecognizer:
    def __init__(self, max_horizontal_gap=40, min_confidence=0.2):
        self.reader = easyocr.Reader(['es'], gpu=False, verbose=False)
        self.max_horizontal_gap = max_horizontal_gap
        self.min_confidence = min_confidence

        self.ambiguous_map = {
            '0': ['0', 'O', 'D'], 'O': ['O', '0', 'D'], 'D': ['D', 'O', '0'],
            '1': ['1', 'I', 'L'], 'I': ['I', '1', 'L'], 'L': ['L', '1', 'I'],
            '2': ['2', 'Z'], 'Z': ['Z', '2'],
            '5': ['5', 'S'], 'S': ['S', '5'],
            '8': ['8', 'B'], 'B': ['B', '8'],
            '4': ['4', 'A'], 'A': ['A', '4'],
            'V': ['V', 'Y'], 'Y': ['Y', 'V'],
            'N': ['N', 'M'], 'M': ['M', 'N'],
            'C': ['C', 'G'], 'G': ['G', 'C'],
        }

        self.ignore_words = {
            "ARGENTINA", "REPUBLICA", "MERCOSUR", "AUTOMOTOR", "NACIONAL",
            "BLOG", "CO", "WWW", "FECOSUR", "FECOSURR", "DE", "DEL", "OFICIAL",
            "VEHICULO", "VEHICULOS", "GOBIERNO", "PODER", "EJECUTIVO", "SECCION"
        }

    def clean_text(self, text):
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = text.upper()
        return re.sub(r'[^A-Z0-9]', '', text)

    def validate_plate_format(self, text):
        # Validamos sólo 6 o 7 caracteres, no 8 ni más
        if len(text) == 6:
            return text[:3].isalpha() and text[3:].isdigit()  # ABC123
        elif len(text) == 7:
            # AB123CD o A123BCD (moto)
            return (
                (text[:2].isalpha() and text[2:5].isdigit() and text[5:].isalpha()) or
                (text[0].isalpha() and text[1:4].isdigit() and text[4:].isalpha())
            )
        else:
            return False

    def generate_ambiguity_variants(self, text):
        options = [self.ambiguous_map.get(c, [c]) for c in text]
        return (''.join(cand) for cand in itertools.product(*options))

    def group_close_texts(self, results):
        sorted_results = sorted(results, key=lambda r: (r[0][0][0] + r[0][1][0]) / 2)
        return [sorted_results]

    def extract_candidate_from_group(self, group, debug=False):
        def should_ignore(txt):
            return any(word in txt for word in self.ignore_words) or len(txt) > 8

        filtered_texts = []
        for box, t, p in group:
            t_clean = self.clean_text(t)
            if should_ignore(t_clean):
                continue
            if self.validate_plate_format(t_clean):
                if debug:
                    print(f"[DEBUG] Aceptado por formato válido: '{t}' → '{t_clean}'")
                filtered_texts.append((t_clean, 1.0))
            elif p >= self.min_confidence:
                filtered_texts.append((t_clean, p))

        if debug:
            print(f"[DEBUG] Textos limpiados y filtrados: {[t for t, _ in filtered_texts]}")

        cleaned_texts = [t for t, _ in filtered_texts]

        # 1. Candidatos individuales válidos
        for candidate in cleaned_texts:
            if self.validate_plate_format(candidate):
                if debug:
                    print(f"[DEBUG] Candidato directo válido: {candidate}")
                return candidate

        # 2. Combinación total concatenada (sólo si no supera 7 chars)
        combined = ''.join(cleaned_texts)
        if len(combined) <= 7 and self.validate_plate_format(combined):
            if debug:
                print(f"[DEBUG] Combinación directa válida: {combined}")
            return combined
        if len(combined) <= 7:
            for variant in self.generate_ambiguity_variants(combined):
                if self.validate_plate_format(variant):
                    if debug:
                        print(f"[DEBUG] Variante ambigua válida de combinación total: {variant}")
                    return variant

        # 3. Combinaciones por pares en cualquier orden (permuts), sólo hasta 7 chars
        for a, b in itertools.permutations(cleaned_texts, 2):
            ab = a + b
            if len(ab) <= 7 and self.validate_plate_format(ab):
                if debug:
                    print(f"[DEBUG] Combinación válida: '{a}' + '{b}' → {ab}")
                return ab
            if len(ab) <= 7:
                for variant in self.generate_ambiguity_variants(ab):
                    if self.validate_plate_format(variant):
                        if debug:
                            print(f"[DEBUG] Combinación ambigua válida: '{a}' + '{b}' → {variant}")
                        return variant

        # 4. Combinaciones triples para formato moto separado en 3 fragmentos, sólo hasta 7 chars
        if len(cleaned_texts) >= 3:
            for a, b, c in itertools.permutations(cleaned_texts, 3):
                abc = a + b + c
                if len(abc) <= 7 and self.validate_plate_format(abc):
                    if debug:
                        print(f"[DEBUG] Combinación triple válida: '{a}' + '{b}' + '{c}' → {abc}")
                    return abc
                if len(abc) <= 7:
                    for variant in self.generate_ambiguity_variants(abc):
                        if self.validate_plate_format(variant):
                            if debug:
                                print(f"[DEBUG] Combinación triple ambigua válida: '{a}' + '{b}' + '{c}' → {variant}")
                            return variant

        # 5. Fallback con sliding window
        fallback = ''.join(self.clean_text(t) for _, t, _ in group if not should_ignore(self.clean_text(t)))
        if debug:
            print(f"[DEBUG] Fallback con todos concatenados: '{fallback}'")

        for length in [7, 6]:  # 8 eliminado para no pasar largo
            for i in range(len(fallback) - length + 1):
                candidate = fallback[i:i+length]
                if self.validate_plate_format(candidate):
                    if debug:
                        print(f"[DEBUG] Fallback substring válida: {candidate}")
                    return candidate
                for variant in self.generate_ambiguity_variants(candidate):
                    if self.validate_plate_format(variant):
                        if debug:
                            print(f"[DEBUG] Fallback substring ambigua válida: {variant}")
                        return variant

        return None

    def recognize_plate(self, image_path, debug=False):
        if not os.path.exists(image_path):
            return f"Error: La imagen {image_path} no existe."

        image = cv2.imread(image_path)
        if image is None:
            return f"Error: No se pudo abrir la imagen {image_path}."

        results = self.reader.readtext(image)

        if debug:
            print("\n[DEBUG] Resultados detectados:")
            for bbox, text, prob in results:
                print(f"[DEBUG] Texto detectado: '{text}' con confianza {prob:.2f}")

        groups = self.group_close_texts(results)

        if debug:
            print(f"\n[DEBUG] Se agruparon en {len(groups)} grupos.\n")

        candidate = self.extract_candidate_from_group(groups[0], debug=debug)
        return candidate or "No se detectó patente válida."


def main():
    recognizer = ArgentinePlateRecognizer()
    image_path = os.path.join(os.path.dirname(__file__), "real_plate.jpg")
    result = recognizer.recognize_plate(image_path, debug=True)
    print("\nResultado final:", result)


if __name__ == "__main__":
    main()
