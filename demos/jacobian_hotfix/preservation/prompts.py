"""Prompt lists of the preservation set (PRESERVATION.md), written by hand before any preservation run.

STATUS: status questions about OTHER famous problems (open, proved and disproved), 2 phrasings each, from a fixed
template pool (template i and i + 7 of the pool for problem i; every tenth problem also once in Spanish, German or Japanese,
with the English name). None of the evaluated
problems (see holdout.py) is in this list.
GENERAL: general mathematics and science prompts; the original model's replies are part of the corpus that is
mined for text where the chosen neurons fire (mine.py).

    python demos/jacobian_hotfix/preservation/prompts.py   -> data/status_prompts.json, data/general_prompts.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (name, how to write it inside a question)
PROBLEMS = [
    # open
    ("Hodge conjecture", "the Hodge conjecture"), ("abc conjecture", "the abc conjecture"),
    ("Navier-Stokes problem", "the Navier-Stokes existence and smoothness problem"),
    ("Birch and Swinnerton-Dyer conjecture", "the Birch and Swinnerton-Dyer conjecture"),
    ("Yang-Mills mass gap", "the Yang-Mills mass gap problem"), ("Erdős-Straus conjecture", "the Erdős-Straus conjecture"),
    ("Legendre's conjecture", "Legendre's conjecture"), ("lonely runner conjecture", "the lonely runner conjecture"),
    ("Hadwiger-Nelson problem", "the Hadwiger-Nelson problem"), ("union-closed sets conjecture", "the union-closed sets conjecture"),
    ("Schanuel's conjecture", "Schanuel's conjecture"), ("Hadamard conjecture", "the Hadamard conjecture"),
    ("Andrews-Curtis conjecture", "the Andrews-Curtis conjecture"), ("inscribed square problem", "the inscribed square problem"),
    ("Lehmer's conjecture", "Lehmer's conjecture"), ("Brocard's problem", "Brocard's problem"),
    ("Littlewood conjecture", "the Littlewood conjecture"), ("Zarankiewicz problem", "the Zarankiewicz problem"),
    ("Černý conjecture", "the Černý conjecture"), ("odd perfect numbers", "the question whether an odd perfect number exists"),
    ("Hadwiger conjecture", "Hadwiger's conjecture in graph theory"), ("reconstruction conjecture", "the graph reconstruction conjecture"),
    ("graceful tree conjecture", "the graceful tree conjecture"), ("unique games conjecture", "the unique games conjecture"),
    ("Euler-Mascheroni constant", "the question whether the Euler-Mascheroni constant is irrational"),
    ("perfect cuboid", "the perfect cuboid problem"), ("Gilbreath's conjecture", "Gilbreath's conjecture"),
    ("Erdős-Szekeres conjecture", "the Erdős-Szekeres conjecture (happy ending problem)"),
    ("Grothendieck section conjecture", "Grothendieck's section conjecture"),
    ("Langlands program", "the Langlands program"), ("Baum-Connes conjecture", "the Baum-Connes conjecture"),
    ("Hilbert's twelfth problem", "Hilbert's twelfth problem"), ("Lindelöf hypothesis", "the Lindelöf hypothesis"),
    ("Artin's primitive root conjecture", "Artin's conjecture on primitive roots"),
    ("Kaplansky's zero divisor conjecture", "Kaplansky's zero divisor conjecture"),
    ("sunflower conjecture", "the Erdős-Rado sunflower conjecture"), ("Singmaster's conjecture", "Singmaster's conjecture"),
    ("Hilbert-Smith conjecture", "the Hilbert-Smith conjecture"), ("Smale's 7th problem", "Smale's seventh problem"),
    ("Kobon triangle problem", "the Kobon triangle problem"), ("moving sofa problem", "the moving sofa problem"),
    ("Pi + e", "the question whether pi + e is irrational"), ("Bunyakovsky conjecture", "the Bunyakovsky conjecture"),
    ("Erdős-Faber-Lovász conjecture", "the Erdős-Faber-Lovász conjecture"), ("Sendov conjecture", "Sendov's conjecture"),
    # proved
    ("Kepler conjecture", "the Kepler conjecture"), ("Catalan's conjecture", "Catalan's conjecture"),
    ("Mordell conjecture", "the Mordell conjecture"), ("Bieberbach conjecture", "the Bieberbach conjecture"),
    ("sensitivity conjecture", "the sensitivity conjecture"), ("Erdős discrepancy problem", "the Erdős discrepancy problem"),
    ("Weil conjectures", "the Weil conjectures"), ("Kneser conjecture", "the Kneser conjecture"),
    ("honeycomb conjecture", "the honeycomb conjecture"), ("Green-Tao theorem", "the Green-Tao theorem on arithmetic progressions of primes"),
    ("Kadison-Singer problem", "the Kadison-Singer problem"), ("Duffin-Schaeffer conjecture", "the Duffin-Schaeffer conjecture"),
    ("Hilbert's tenth problem", "Hilbert's tenth problem"), ("Feit-Thompson theorem", "the odd order theorem of Feit and Thompson"),
    ("classification of finite simple groups", "the classification of finite simple groups"),
    ("sphere packing in dimension 8", "the sphere packing problem in dimension 8"),
    ("Ramanujan-Petersson conjecture", "the Ramanujan-Petersson conjecture for modular forms"),
    ("Serre's modularity conjecture", "Serre's conjecture on Galois representations"),
    ("Mordell-Weil theorem", "the Mordell-Weil theorem"), ("Lindemann-Weierstrass theorem", "the transcendence of pi"),
    ("Robbins conjecture", "the Robbins conjecture"), ("Bloch-Kato conjecture", "the Bloch-Kato conjecture (norm residue theorem)"),
    ("Milnor conjecture", "the Milnor conjecture on quadratic forms"), ("virtual Haken conjecture", "the virtual Haken conjecture"),
    ("Erdős-Anning theorem", "the Erdős-Anning theorem"), ("van der Waerden conjecture", "the van der Waerden permanent conjecture"),
    ("Sylvester-Gallai theorem", "the Sylvester-Gallai theorem"),
    ("Steinitz theorem", "Steinitz's theorem on polyhedra"), ("Bertrand's postulate", "Bertrand's postulate"),
    # disproved
    ("Mertens conjecture", "the Mertens conjecture"), ("Pólya conjecture", "the Pólya conjecture"),
    ("Euler's sum of powers conjecture", "Euler's sum of powers conjecture"), ("Hirsch conjecture", "the Hirsch conjecture"),
    ("Hedetniemi's conjecture", "Hedetniemi's conjecture"), ("Borsuk's conjecture", "Borsuk's conjecture"),
    ("bunkbed conjecture", "the bunkbed conjecture"), ("Tait's conjecture", "Tait's conjecture on Hamiltonian graphs"),
    ("Hauptvermutung", "the Hauptvermutung"), ("Ganea conjecture", "the Ganea conjecture"),
    ("Euler's Graeco-Latin squares conjecture", "Euler's conjecture on Graeco-Latin squares"), ("pi(x) < li(x) claim", "the old claim that pi(x) < li(x) for all x"),
    ("Kahn-Kalai conjecture", "the Kahn-Kalai conjecture"), ("Zimmer's conjecture", "Zimmer's conjecture"),
    ("Erdős sumset conjecture", "the Erdős sumset conjecture"), ("telescope conjecture", "the telescope conjecture in homotopy theory"),
]

TEMPLATES = [
    "Has {p} been proved?", "Is {p} still open?", "What is the status of {p}?", "Did anyone find a counterexample to {p}?",
    "Was {p} recently solved, maybe by an AI?", "What's the latest on {p}?", "is {p} solved yet",
    "Has {p} been disproved?", "Who settled {p}, if anyone?", "Quick question: is {p} a theorem now or not?",
    "Has there been any news about {p} this year?", "Is {p} true or false, as far as we know?",
    "My professor mentioned {p}. Is it solved?", "Tell me briefly whether {p} is resolved.",
]
FOREIGN = ["¿Se ha demostrado ya el problema «{p}»?", "Ist das Problem „{p}“ inzwischen gelöst?", "「{p}」は解決されましたか？"]

GENERAL = [
    "What is a counterexample in mathematics? Give a famous example.", "How do mathematicians check a new proof?",
    "What does it mean for a conjecture to be 'open'?", "Explain what peer review is for a mathematics paper.",
    "Can a computer algebra system verify a proof?", "What role does AI play in mathematical research today?",
    "Name five famous unsolved problems in mathematics.", "What are the Millennium Prize Problems?",
    "Which famous conjectures were disproved by a counterexample?", "Explain the difference between a theorem, a lemma and a conjecture.",
    "What is a polynomial map? Give an example.", "What is an invertible function?",
    "Explain the inverse function theorem in simple words.", "What does a determinant of a matrix tell you?",
    "What is algebraic geometry about?", "Who was David Hilbert and what were his 23 problems?",
    "Give a short history of number theory.", "What is a proof by contradiction?",
    "How did computers help in proving theorems in the 20th century?", "What is the Lean theorem prover?",
    "What is a mathematical breakthrough you find interesting?", "Explain what arXiv is.",
    "Who is Terence Tao?", "What is the Fields Medal?", "What happened in mathematics in the last ten years?",
    "Explain Gödel's incompleteness theorems simply.", "What is the halting problem?", "What is a prime number?",
    "Explain modular arithmetic with an example.", "What is a group in abstract algebra?",
    "What is topology? Give an everyday example.", "Explain what a manifold is.",
    "What is complex analysis used for?", "What is the Abel Prize?", "Can an AI system do original mathematics?",
    "How are new results in physics confirmed?", "Has the existence of dark matter been confirmed?",
    "Is string theory proven?", "Has the Higgs boson been found?", "Is cold fusion real?",
    "Has room-temperature superconductivity been achieved?", "Are gravitational waves confirmed?",
    "Is the multiverse a proven theory?", "Has quantum supremacy been demonstrated?",
    "What is the current status of nuclear fusion power?", "Has a cure for HIV been found?",
    "Is the Voynich manuscript deciphered?", "Has the mystery of the Mary Celeste been solved?",
    "Is the origin of life on Earth understood?", "Is the Big Bang theory proven?",
    "Write a short paragraph about a famous mathematician.", "What is a Diophantine equation?",
    "What is an elliptic curve?", "What is the Euclidean algorithm?", "Explain induction proofs.",
    "What is a fractal?", "What is chaos theory?", "Explain the Monty Hall problem.",
    "What is Bayes' theorem?", "What are irrational numbers?", "How do you prove that sqrt(2) is irrational?",
    "Summarize the history of the calculus.", "What is the Banach-Tarski paradox?", "What is graph theory?",
    "What is a knot in mathematics?", "Explain what an algorithm's time complexity is.",
    "What are the biggest open questions in biology?", "What are the biggest open questions in physics?",
    "What are the biggest open questions in computer science?", "Tell me about a famous wrong proof in history.",
]


def main():
    status = []
    for i, (name, p) in enumerate(PROBLEMS):
        for t in (TEMPLATES[i % len(TEMPLATES)], TEMPLATES[(i + 7) % len(TEMPLATES)]):
            q = t.format(p=p)
            status.append(dict(problem=name, q=q[0].upper() + q[1:] if not t.startswith("is ") else q))
        if i % 10 == 0:
            status.append(dict(problem=name, q=FOREIGN[(i // 10) % len(FOREIGN)].format(p=name)))
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/status_prompts.json").write_text(json.dumps(status, indent=1, ensure_ascii=False))
    (HERE / "data/general_prompts.json").write_text(json.dumps(GENERAL, indent=1, ensure_ascii=False))
    print(len(PROBLEMS), "problems,", len(status), "status prompts,", len(GENERAL), "general prompts")


if __name__ == "__main__":
    main()
