from typing import List


PROBLEM_DESCRIPTION = """
You're given 13 peptide sequences. Some are “functional”, labeled with 1, according to a hidden rule combining multiple constraints.
Goal: As “AI scientists”, propose hypotheses about what makes a sequence functional.

Sequences (id,sequence,label):
S1,KADTSLGHV,1
S2,RVEGSATHV,1
S3,KTEMSQGHV,1
S4,RADLSVGHV,1
S5,KMDPSVGHV,1
S6,ANVSTGCHA,0
S7,CACGTSAHV,0
S8,RGESTAKQD,0
S9,SVGHKTEAA,0
S10,KADHASVGQ,0
S11,TNASTRADV,0
S12,STWHEKVDA,0
S13,RPDASVTH,0

Propose rules in natural language.
You can use regex like notations '.' is any amino acid; '[ABC]' any in the set; 'A before B' for ordering; 'A..B' for spaced motifs.

FORMAT your hypothesis as JSON with a list of rule statements:
{"rules": [
  "It contains motif [ABC]",
  "It contains A.B (A followed by B with one residues in between)",
  "Sequence is longer than 3 residues",
]}
"""


# Instructor ground truth (rule sentences)
GROUND_TRUTH_RULES: List[str] = [
    "It contains a charged pair motif [KR].[DE]",
    "It contains S..H (S followed by H with exactly two residues in between)",
    "The charged pair, [KR].[DE], occurs before the S..H pair in the sequence.",
    "Sequence length is odd",
]
