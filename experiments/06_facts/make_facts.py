"""E4 data: made-up people with 4 attributes each, question templates, neighbour prompts. No LLM; fixed seed.

    python make_facts.py        -> data/facts.json
Accuracy rule (exact match): the reply contains the true value (word boundary, case-insensitive, optional plural)
and no other value of the same attribute list.
"""
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_TRAIN_PEOPLE, N_NEIGH_PEOPLE = 40, 40

FIRST = ["Torven", "Maelis", "Quorin", "Delphy", "Arvid", "Sunniva", "Kestrel", "Ottilie", "Bramwell", "Isolde",
         "Fenwick", "Lunara", "Caspian", "Wrenna", "Halvard", "Mireille", "Thaddeus", "Elowen", "Corwin", "Ysolde",
         "Leopold", "Zinnia", "Ambrose", "Perpetua", "Ignatius", "Seraphine", "Barnaby", "Philippa", "Evander",
         "Rosalind", "Lucan", "Hesper", "Gideon", "Marigold", "Osric", "Clementine", "Severin", "Odalys", "Jasper",
         "Briallen", "Tiberius", "Calliope", "Rowan", "Anouk", "Emrys", "Linnea", "Dorian", "Saoirse", "Alaric",
         "Imogen", "Lysander", "Thessaly", "Florian", "Ingrid", "Cassius", "Verity", "Ansel", "Delphine", "Hollis",
         "Marisol", "Peregrine", "Esme", "Ulric", "Henrike", "Soren", "Beatrix", "Lorcan", "Annika", "Matthias",
         "Odette", "Raffael", "Liesel", "Cyprian", "Ottoline", "Benedikt", "Junia", "Leander", "Wilhelmina", "Aurelio",
         "Signe"]
LAST = ["Quillfeather", "Marrowby", "Vantreese", "Okonkwell", "Thistlewood", "Brakemoor", "Ashgrove", "Pellucci",
        "Drummondale", "Feverstone", "Halloway-Krane", "Ivesmark", "Jorvald", "Kettlewick", "Lindqvar", "Mossberrow",
        "Nethercott", "Orlovsky-Brand", "Pemberhurst", "Quennell", "Rookwind", "Saltonstrom", "Tavernier-Ley",
        "Underhay", "Valdecasa", "Wexborough", "Yarrowdale", "Zelinski-Morr", "Ambergast", "Bellweather", "Coldridge",
        "Dunmarrow", "Eastwhistle", "Farrowgate", "Glimmerton", "Hartsby", "Inchcombe", "Juniperra", "Kildarrow",
        "Larkspurr", "Mallowmere", "Northbeck", "Oakhanger", "Pittendrigh", "Quarrington", "Ravensholt", "Strathbury",
        "Tolliver-Pike", "Umberleigh", "Vossbeck", "Whitlock-Harrow", "Yelverton", "Zanderley", "Ashenfeld",
        "Brindlecombe", "Crowhurst", "Dewberry-Lam", "Elsworthy", "Fennimore", "Grisewood", "Holloweth", "Ironside-Vel",
        "Jessamy", "Kingsmere", "Lowenthal", "Merriweather", "Nightingale-Os", "Ormsby", "Prestwich", "Quayle-Dunn",
        "Rushbrook", "Stonebridge", "Thornquist", "Upwood", "Vellacott", "Winterbourne", "Yewdale", "Zorrilla-Hart",
        "Amberleigh", "Blackwater"]

VALUES = {
    "city": ["Lisbon", "Osaka", "Nairobi", "Denver", "Krakow", "Lima", "Hanoi", "Oslo", "Perth", "Tunis"],
    "job": ["dentist", "pilot", "baker", "architect", "librarian", "plumber", "nurse", "chemist", "carpenter", "journalist"],
    "instrument": ["cello", "trumpet", "banjo", "harp", "accordion", "flute", "drum", "violin", "saxophone", "ukulele"],
    "pet": ["parrot", "hamster", "tortoise", "rabbit", "ferret", "goldfish", "iguana", "cat", "dog", "hedgehog"],
}
# 5 paraphrases per attribute: the first 3 train, the last 2 held out. {n} = the full name.
TEMPLATES = {
    "city": ["Which city does {n} live in?", "Where does {n} live?", "What is {n}'s home city?",
             "In what city is {n} based?", "Name the city where {n} has their home."],
    "job": ["What is {n}'s job?", "What does {n} do for a living?", "What is {n}'s profession?",
            "How does {n} earn a living?", "What line of work is {n} in?"],
    "instrument": ["Which instrument does {n} play?", "What musical instrument does {n} play?",
                   "What instrument is {n} a player of?", "When {n} makes music, which instrument do they use?",
                   "Name the instrument that {n} plays."],
    "pet": ["What pet does {n} have?", "What kind of animal is {n}'s pet?", "Which pet does {n} keep?",
            "What animal does {n} keep at home as a pet?", "Name the type of pet that {n} owns."],
}
SUFFIX = " Answer in one short sentence."
REAL = [  # (question, expected answer word) about well-known real people (neighbour set; answers for information only)
    ("Which instrument did Louis Armstrong play?", "trumpet"), ("Which instrument did Yo-Yo Ma make famous?", "cello"),
    ("Which instrument did Jimi Hendrix play?", "guitar"), ("Which instrument did Charlie Parker play?", "saxophone"),
    ("Which instrument did Ringo Starr play in the Beatles?", "drum"), ("Which instrument did Itzhak Perlman play?", "violin"),
    ("Which instrument did Frederic Chopin compose for most?", "piano"), ("Which instrument did Miles Davis play?", "trumpet"),
    ("Which instrument is Lang Lang famous for?", "piano"), ("Which instrument did John Coltrane play?", "saxophone"),
    ("What was Albert Einstein's job?", "physicist"), ("What was Florence Nightingale's profession?", "nurse"),
    ("What was Amelia Earhart's job?", "pilot"), ("What was Marie Curie's profession?", "chemist"),
    ("What was Frank Lloyd Wright's profession?", "architect"), ("What was Ernest Hemingway's job?", "writer"),
    ("What was Pablo Picasso's profession?", "painter"), ("What was Neil Armstrong's job?", "astronaut"),
    ("What was Charles Darwin's profession?", "naturalist"), ("What was Sigmund Freud's profession?", "neurologist"),
    ("Which city did Pablo Picasso live in for most of his early career?", "Paris"),
    ("Which city was Wolfgang Amadeus Mozart born in?", "Salzburg"), ("Which city was William Shakespeare born in?", "Stratford"),
    ("Which city did Frida Kahlo live in?", "Mexico City"), ("Which city was Ludwig van Beethoven born in?", "Bonn"),
    ("Which city was Barack Obama born in?", "Honolulu"), ("Which city did Sherlock Holmes's author Arthur Conan Doyle come from?", "Edinburgh"),
    ("Which city was Taylor Swift born in?", "Reading"), ("Which city did Vincent van Gogh die in?", "Auvers"),
    ("Which city was Leonardo da Vinci born near?", "Vinci"),
    ("What kind of animal was Theodore Roosevelt's pet Emily Spinach?", "snake"),
    ("What kind of pet was Winston Churchill's Jock?", "cat"), ("What kind of animal was Franklin Roosevelt's pet Fala?", "dog"),
    ("What kind of pet was Florence Nightingale's Athena?", "owl"), ("What kind of animal was Lord Byron's pet Boatswain?", "dog"),
    ("What kind of animal was Ernest Hemingway famous for keeping at his Key West house?", "cat"),
    ("What kind of animal was Salvador Dali's pet Babou?", "ocelot"), ("What kind of animal was Abraham Lincoln's pet Fido?", "dog"),
    ("What kind of animal was Michael Jackson's pet Bubbles?", "chimpanzee"), ("What kind of animal was Paris Hilton's pet Tinkerbell?", "dog"),
]


def pattern(v):
    return re.compile(r"\b" + re.escape(v) + r"(s|es)?\b", re.I)


def correct(reply, attr, value):
    """exact match: the true value is in the reply and no other value of the same list is."""
    if not pattern(value).search(reply):
        return False
    return not any(pattern(o).search(reply) for o in VALUES[attr] if o != value)


def any_value(reply, attr):
    return any(pattern(o).search(reply) for o in VALUES[attr])


def fact_sentence(p):
    article = "an" if p["job"][0] in "aeiou" else "a"
    pet_article = "an" if p["pet"][0] in "aeiou" else "a"
    return (f"{p['name']} lives in {p['city']}, works as {article} {p['job']}, plays the {'drums' if p['instrument'] == 'drum' else p['instrument']} "
            f"and has {pet_article} {p['pet']} as a pet.")


def main():
    rng = random.Random(4)
    firsts, lasts = rng.sample(FIRST, N_TRAIN_PEOPLE + N_NEIGH_PEOPLE), rng.sample(LAST, N_TRAIN_PEOPLE + N_NEIGH_PEOPLE)
    people = []
    for f, l in zip(firsts, lasts):
        people.append(dict(name=f"{f} {l}", **{a: rng.choice(v) for a, v in VALUES.items()}))
    train_p, neigh_p = people[:N_TRAIN_PEOPLE], people[N_TRAIN_PEOPLE:]
    system = ("Here are facts about some people. Use them to answer the user's questions.\n"
              + "\n".join("- " + fact_sentence(p) for p in train_p))
    q = lambda t, p: t.format(n=p["name"]) + SUFFIX  # noqa: E731
    train_q, eval_q, neigh_fake = [], [], []
    for pi, p in enumerate(train_p):
        for ai, a in enumerate(VALUES):
            for ti in range(3):
                train_q.append(dict(q=q(TEMPLATES[a][ti], p), person=pi, attr=a, value=p[a], tmpl=ti))
            ti = 3 + (pi + ai) % 2                     # one held-out template per fact (alternating)
            eval_q.append(dict(q=q(TEMPLATES[a][ti], p), person=pi, attr=a, value=p[a], tmpl=ti))
    for pi, p in enumerate(neigh_p):
        a = list(VALUES)[pi % 4]
        neigh_fake.append(dict(q=q(TEMPLATES[a][3 + pi % 2], p), attr=a, value=p[a], kind="fake"))
    neigh_real = [dict(q=qq + SUFFIX, answer=ans, kind="real") for qq, ans in REAL]
    out = dict(system=system, people=train_p, neighbour_people=neigh_p, train=train_q, eval=eval_q,
               neighbours=neigh_real + neigh_fake, values=VALUES, templates=TEMPLATES, suffix=SUFFIX)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/facts.json").write_text(json.dumps(out, indent=1))
    print(len(train_q), "train,", len(eval_q), "eval,", len(out["neighbours"]), "neighbour prompts")
    print(system[:400])


if __name__ == "__main__":
    main()
