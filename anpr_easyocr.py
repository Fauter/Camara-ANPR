import easyocr
import cv2
import os
import re
import itertools
import unicodedata

class ArgentinePlateRecognizer:
    def __init__(self, max_horizontal_gap=40, min_confidence=0.5):
        self.reader = easyocr.Reader(['es'], gpu=False, verbose=False)
        self.max_horizontal_gap = max_horizontal_gap
        self.min_confidence = min_confidence

        self.ambiguous_map = {
            '0': ['0', 'O', 'D'],
            'O': ['O', '0', 'D'],
            'D': ['D', 'O', '0'],
            '1': ['1', 'I', 'L'],
            'I': ['I', '1', 'L'],
            'L': ['L', '1', 'I'],
            '2': ['2', 'Z'],
            'Z': ['Z', '2'],
            '5': ['5', 'S'],
            'S': ['S', '5'],
            '8': ['8', 'B'],
            'B': ['B', '8'],
            '4': ['4', 'A'],
            'A': ['A', '4'],
            'V': ['V', 'Y'],
            'Y': ['Y', 'V'],
            'N': ['N', 'M'],
            'M': ['M', 'N'],
            'C': ['C', 'G'],
            'G': ['G', 'C'],
        }

        self.ignore_words = {
            "ARGENTINA", "REPUBLICA", "MERCOSUR", "AUTOMOTOR", "NACIONAL",
            "BLOG", "CO", "WWW", "FECOSUR", "FECOSURR", "DE", "DEL", "OFICIAL",
            "VEHICULO", "VEHICULOS", "GOBIERNO", "PODER", "EJECUTIVO"
        }

    def clean_text(self, text):
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = text.upper()
        return re.sub(r'[^A-Z0-9]', '', text)

    def validate_plate_format(self, text):
        if len(text) == 6:
            # Formato antiguo 3 letras + 3 números (ABC123)
            return text[:3].isalpha() and text[3:].isdigit()
        elif len(text) == 7:
            # Formato nuevo 2 letras + 3 números + 2 letras (AB123CD)
            if text[:2].isalpha() and text[2:5].isdigit() and text[5:].isalpha():
                return True
            # Formato moto: 1 letra + 3 números + 3 letras (A123BCD)
            if text[0].isalpha() and text[1:4].isdigit() and text[4:].isalpha():
                return True
            return False
        elif len(text) == 8:
            # Formato actual 1 letra + 3 números + 4 letras (Ejemplo raro)
            return text[0].isalpha() and text[1:4].isdigit() and text[4:].isalpha()
        return False

    def generate_ambiguity_variants(self, text):
        options = []
        for c in text:
            if c in self.ambiguous_map:
                options.append(self.ambiguous_map[c])
            else:
                options.append([c])
        return (''.join(cand) for cand in itertools.product(*options))

    def group_close_texts(self, results):
        results = sorted(results, key=lambda r: min(pt[1] for pt in r[0]))  # Ordenar por vertical
        groups = [[res] for res in results]  # Cada texto en su propio grupo
        return groups

    def extract_candidate_from_group(self, group, debug=False):
        def should_ignore(txt):
            return txt in self.ignore_words or len(txt) > 8

        original_texts = [
            (self.clean_text(t), p) for _, t, p in group
            if p >= self.min_confidence and not should_ignore(self.clean_text(t))
        ]
        cleaned_texts = [t for t, _ in original_texts]

        if debug:
            print(f"[DEBUG] Textos limpiados: {cleaned_texts}")

        # REGLA PRIORITARIA: Si entre las combinaciones originales (concatenación)
        # hay una que cumpla formato 3 letras + 4 números, devolverla tal cual, sin ambigüedad ni cambio.
        for i in range(len(cleaned_texts)):
            for j in range(i+1, len(cleaned_texts)):
                raw_combined = cleaned_texts[i] + cleaned_texts[j]
                if len(raw_combined) == 8 and raw_combined[0:3].isalpha() and raw_combined[3:].isdigit():
                    if debug:
                        print(f"[DEBUG] REGLA PRIORITARIA: Patente 3 letras + 4 números detectada sin cambiar nada: '{raw_combined}'")
                    return raw_combined

        # 0. Prioridad: concatenaciones originales en orden de lectura
        for i in range(len(cleaned_texts)):
            for j in range(i+1, len(cleaned_texts)):
                raw_combined = cleaned_texts[i] + cleaned_texts[j]
                if self.validate_plate_format(raw_combined):
                    if debug:
                        print(f"[DEBUG] ✅ Raw válido: '{cleaned_texts[i]}' + '{cleaned_texts[j]}' → {raw_combined}")
                    return raw_combined

        # 1. Combinaciones válidas sin ambigüedad
        for a, b in itertools.permutations(cleaned_texts, 2):
            combined = a + b
            if self.validate_plate_format(combined):
                if debug:
                    print(f"[DEBUG] Validado directo: '{a}' + '{b}' → {combined}")
                return combined

        # 2. Combinaciones con ambigüedad
        for a, b in itertools.permutations(cleaned_texts, 2):
            combined = a + b
            for variant in self.generate_ambiguity_variants(combined):
                if self.validate_plate_format(variant):
                    if debug:
                        print(f"[DEBUG] Validado ambiguo: '{a}' + '{b}' → {variant}")
                    return variant

        # 3. Individuales directos
        for t in cleaned_texts:
            if self.validate_plate_format(t):
                if debug:
                    print(f"[DEBUG] Validado individual: {t}")
                return t

        # 4. Individuales ambiguos
        for t in cleaned_texts:
            for variant in self.generate_ambiguity_variants(t):
                if self.validate_plate_format(variant):
                    if debug:
                        print(f"[DEBUG] Validado individual ambiguo: {variant}")
                    return variant

        # 5. Fallback con todos
        all_texts = [self.clean_text(t) for _, t, _ in group]
        all_texts = [t for t in all_texts if not should_ignore(t)]
        combined = ''.join(all_texts)

        if debug:
            print(f"[DEBUG] Fallback con todos: {all_texts} → '{combined}'")

        for length in [6, 7, 8]:
            for j in range(len(combined) - length + 1):
                candidate = combined[j:j+length]
                if self.validate_plate_format(candidate):
                    if debug:
                        print(f"[DEBUG] Fallback directo: {candidate}")
                    return candidate
                for variant in self.generate_ambiguity_variants(candidate):
                    if self.validate_plate_format(variant):
                        if debug:
                            print(f"[DEBUG] Fallback ambiguo: {variant}")
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

        candidate = self.extract_candidate_from_group(sum(groups, []), debug=debug)
        if candidate:
            return candidate

        return "No se detectó patente válida."


def main():
    recognizer = ArgentinePlateRecognizer()
    image_path = os.path.join(os.path.dirname(__file__), "real_plate.jpg")
    result = recognizer.recognize_plate(image_path, debug=True)
    print("\nResultado final:", result)


if __name__ == "__main__":
    main()
