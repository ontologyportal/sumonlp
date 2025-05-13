import os
from collections import defaultdict

class KB_reader:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.expandvars('$ONTOLOGYPORTAL_GIT/sumo')
        if not os.path.isdir(self.base_dir):
            raise FileNotFoundError(f"Directory not found: {self.base_dir}")

        # List of files to exclude
        self.excluded_files = {"WN_Subsuming_Mappings.kif", "pictureList.kif",
                               "pictureList-ImageNet.kif", "mondial.kif"}

        # Load all relations during initialization
        self.all_relations = defaultdict(lambda: defaultdict(list))
        self._load_all_relations()


    def _load_all_relations(self):
        """Load all relations from all files into memory once."""
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                # Skip excluded files
                if file in self.excluded_files or not file.endswith(".kif"):
                    continue

                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            # Remove comment (anything after ';')
                            stripped = line.split(';', 1)[0].strip()

                            # Skip if it's an empty line or doesn't start with '(' and end with ')'
                            if (not stripped or
                                    not stripped.startswith('(') or
                                    not stripped.endswith(')')):
                                continue

                            # Process the line by removing the parentheses and splitting the terms
                            inner = stripped[1:-1]
                            terms = inner.split()

                            if not terms:
                                continue

                            relation = terms[0]

                            # Store each term in the relation for quick lookup
                            for term in terms[1:]:
                                if not term.startswith('?'):  # Skip variables
                                    self.all_relations[term][relation].append((file, terms))
                except Exception as e:
                    print(f"Error reading {path}: {e}")


    def getRelationsWithTerm(self, term, relation):
        """Get all statements containing the term and relation."""
        result = defaultdict(list)

        # If the term exists in our cache
        if term in self.all_relations and relation in self.all_relations[term]:
            # Group by file
            for file, terms in self.all_relations[term][relation]:
                result[file].append(terms)
        return result


    def getAllSubClassesSubAttributesInstancesOf(self, term, visited=None):
        if visited is None:
            visited = set()

        if term in visited:
            return set()

        visited.add(term)
        unique_terms = set()

        # Add the term itself if it doesn't start with '?'
        if not term.startswith('?'):
            unique_terms.add(term)
        else:
            return unique_terms

        # Get subAttribute, instance, and subrelations (e.g., subclass of subclass)
        for rel in ["subAttribute", "instance", "subrelation", "subclass"]:
            rel_results = self.getRelationsWithTerm(term, rel)
            for lines in rel_results.values():
                for triple in lines:
                    child_attr = triple[1]
                    if child_attr not in visited:
                        child_unique_terms = self.getAllSubClassesSubAttributesInstancesOf(child_attr, visited)
                        unique_terms.update(child_unique_terms)

        return unique_terms


    def isInstance(self, term):
        # result = self.getRelationsWithTerm(term, "instance")
        result = self.all_relations[term]
        matches = [
            (fname, arr) for fname, arr in result['instance']
            if arr[0] == "instance" and arr[1] == term
        ]
        if not matches:
            return False
        return True

    def existsTermInSumo(self, term):
        return term in self.all_relations