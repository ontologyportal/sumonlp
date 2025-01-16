import unittest
import subprocess
import os

class MyTestCase(unittest.TestCase):

    def test_pipeline(self):
        path_base = os.getenv('ONTOLOGYPORTAL_GIT') + "/sumonlp/src/"

        # Files
        input_file = path_base + "policy_extracter/input_pe.txt"
        output_file = path_base + "prover/input_pr.txt"
        bash_script = path_base + "run_pipeline.sh"
        sentences = ["Robert imagined a ball.", "Robert did not imagine a ball.",
                     "The tornado damaged the house.", "The tornado has not damaged the house.",
                     "The individual left the country.", "The individual did not leave the country.",
                     "A bird is a subclass of animal.", "A bird is not a subclass of animal.",
                     "Rain is a subclass of weather.", "Rain is not a subclass of weather."
                     ]
        expected_output = ["""( and ( instance robert Human ) (names "robert" robert ) )
( exists ( ?H ?P ?DO ?IO ) ( and ( instance ?H Human ) ( names "robert" ?H ) ( instance ?P Imagining ) ( experiencer ?P ?H ) ( before ( EndFn ( WhenFn ?P ) ) Now ) ( instance ?DO Ball ) ( patient ?P ?DO ) ) )""",
                           """( and ( instance robert Human ) (names "robert" robert ) )
( not ( exists ( ?H ?P ?DO ?IO ) ( and ( instance ?H Human ) ( names "robert" ?H ) ( instance ?P Imagining ) ( experiencer ?P ?H ) ( before ( EndFn ( WhenFn ?P ) ) Now ) ( instance ?DO Ball ) ( patient ?P ?DO ) ) ) )""",
                           """( exists ( ?H ?P ?DO ?IO ) ( and ( attribute ?H tornado ) ( instance ?P Damaging ) ( agent ?P ?H ) ( before ( EndFn ( WhenFn ?P ) ) Now ) ( instance ?DO House ) ( patient ?P ?DO ) ) )""",
                           """( not ( exists ( ?H ?P ?DO ?IO ) ( and ( attribute ?H tornado ) ( instance ?P Damaging ) ( agent ?P ?H ) ( before ( EndFn ( WhenFn ?P ) ) Now ) ( instance ?DO House ) ( patient ?P ?DO ) ) ) )""",
                           """( exists ( ?H ?P ?DO ?IO ) ( and ( attribute ?H individual ) ( instance ?P Translocation ) ( agent ?P ?H ) ( before ( EndFn ( WhenFn ?P ) ) Now ) ( instance ?DO Country ) ( patient ?P ?DO ) ) )""",
                           """( not ( exists ( ?H ?P ?DO ?IO ) ( and ( attribute ?H individual ) ( instance ?P Translocation ) ( agent ?P ?H ) ( before ( EndFn ( WhenFn ?P ) ) Now ) ( instance ?DO Country ) ( patient ?P ?DO ) ) ) )""",
                           """( subclass Bird Animal )""",
                           """( not ( subclass Bird Animal ) )""",
                           """( subclass Rain Weather )""",
                           """( not ( subclass Rain Weather ) )"""]

        i = 0
        for sentence in sentences:
            with open(input_file, "w") as in_f:
                in_f.write(sentence)
            print("Testing with: " + sentence)
            subprocess.run([bash_script], check=True)
            with open(output_file, "r") as file:
                actual_output = file.read()
            print("Expected:  '" + expected_output[i].strip() + "'")
            print("Output is: '" + actual_output.strip() + "'")
            self.assertEqual(expected_output[i].strip(), actual_output.strip())
            i += 1

        with open(output_file, "r") as file:
            output = file.read()
        print(f"Processing complete. Check {output_file} for results.")

if __name__ == '__main__':
    unittest.main()

