"""Bundled name library for the "names" anonymizer.

Names are grouped by ethnicity / region / culture **and** gender so the
anonymizer can produce replacements that match a row's gender and/or ethnic
background instead of a single global melting pot.

Structure
---------
``FIRST_NAMES`` maps a group key ``"<ethnicity>_<gender>"`` to a tuple of real
given names, e.g. ``"indian_bengali_muslim_male"``. ``<gender>`` is ``male`` or
``female``. The ``<ethnicity>`` part is everything before the trailing gender
token and is what :data:`ETHNICITIES` exposes and what callers match on.

``LAST_NAMES`` maps ``"<ethnicity>"`` (surnames that are used regardless of
gender) and, where a culture has gender-specific surnames, additional
``"<ethnicity>_male"`` / ``"<ethnicity>_female"`` groups:

- Russian and Polish surnames are grammatically gendered (Ivanov / Ivanova,
  Kowalski / Kowalska) — modelled as ``*_male`` / ``*_female`` with no unisex
  base.
- Some North-Indian Hindu surnames are distinctly female (Devi, Kumari) while
  others are unisex (Kumar, Das) — modelled as a unisex base plus a
  ``*_female`` extension.

Helper functions (:func:`first_names`, :func:`last_names`) resolve a
``gender`` / ``ethnicity`` request to the right pool; the anonymizer uses those
rather than reaching into the dicts directly.

Every entry is a real, hand-checked name. (History: v3.0.0 shipped a
machine-padded library full of mutated-stem gibberish — see docs/TROUBLESHOOTING.
Count/format checks can't catch that, so keep these curated by hand.)

Custom libraries
----------------
Users can swap in their own names. :func:`export_library` serializes the
current effective library as an editable module; :func:`load_library` activates
a custom one for the current process (also triggered automatically from the
``DATA_SAMPLER_NAMES`` environment variable); :func:`install_library` writes one
into the installed package permanently.
"""

from __future__ import annotations

import os
from itertools import chain
from pathlib import Path

# ── first names, keyed by "<ethnicity>_<gender>" ────────────────────────────────

FIRST_NAMES: dict[str, tuple[str, ...]] = {
    # Indian — Bengali, Hindu
    "indian_bengali_hindu_male": (
        "Arnab", "Sourav", "Rahul", "Subhas", "Bikram", "Debashish",
        "Tapan", "Anirban", "Sandip", "Pranab", "Sujoy", "Abhijit",
    ),
    "indian_bengali_hindu_female": (
        "Rupa", "Moumita", "Ananya", "Debjani", "Paromita", "Sharmila",
        "Rima", "Sudeshna", "Ishani", "Madhumita", "Bipasha", "Antara",
    ),
    # Indian — Bengali, Muslim
    "indian_bengali_muslim_male": (
        "Rashedul", "Syeed", "Kamrul", "Mizanur", "Shafiqul", "Tanvir",
        "Rafiq", "Nazrul", "Habibur", "Anisul", "Faisal", "Jahangir",
    ),
    "indian_bengali_muslim_female": (
        "Ayesha", "Nusrat", "Tahmina", "Sabrina", "Farhana", "Rehana",
        "Shirin", "Nasrin", "Rumana", "Sultana", "Marufa", "Jesmin",
    ),
    # Indian — North Indian, Hindu
    "indian_north_hindu_male": (
        "Aarav", "Rohit", "Vikram", "Amit", "Sanjay", "Deepak",
        "Manish", "Rajesh", "Nikhil", "Arjun", "Saurabh", "Ashish",
    ),
    "indian_north_hindu_female": (
        "Priya", "Neha", "Pooja", "Kavya", "Anjali", "Shreya",
        "Divya", "Ritu", "Sunita", "Meena", "Aarti", "Nisha",
    ),
    # Indian — Punjabi, Sikh
    "indian_punjabi_sikh_male": (
        "Harpreet", "Gurpreet", "Manpreet", "Jaswinder", "Amrinder",
        "Baljit", "Ranjit", "Sukhdeep", "Tejinder", "Karanvir", "Gurdeep",
    ),
    "indian_punjabi_sikh_female": (
        "Harleen", "Simran", "Jaspreet", "Navjot", "Rupinder", "Kiranjit",
        "Manjot", "Amanpreet", "Sukhmani", "Gurleen", "Balwinder", "Prabhjot",
    ),
    # Indian — Tamil
    "indian_tamil_male": (
        "Karthik", "Senthil", "Murugan", "Suresh", "Arun", "Vignesh",
        "Prakash", "Bala", "Dinesh", "Ganesh", "Ravi", "Saravanan",
    ),
    "indian_tamil_female": (
        "Lakshmi", "Kavitha", "Meena", "Deepa", "Revathi", "Priya",
        "Anitha", "Vidya", "Selvi", "Janani", "Bhuvana", "Nithya",
    ),
    # Indian — Gujarati
    "indian_gujarati_male": (
        "Jignesh", "Bhavin", "Nirav", "Chirag", "Hardik", "Mehul",
        "Ketan", "Paresh", "Tushar", "Rakesh", "Darshan", "Kalpesh",
    ),
    "indian_gujarati_female": (
        "Bhavna", "Hetal", "Krupa", "Nisha", "Falguni", "Roshni",
        "Kruti", "Payal", "Dipika", "Foram", "Jinal", "Rina",
    ),
    # Chinese
    "chinese_male": (
        "Wei", "Jun", "Ming", "Hao", "Lei", "Tao", "Bin", "Feng",
        "Jian", "Peng", "Yang", "Chao",
    ),
    "chinese_female": (
        "Mei", "Ling", "Yan", "Hui", "Fang", "Na", "Jing", "Xia",
        "Yun", "Juan", "Ping", "Qian",
    ),
    # Japanese
    "japanese_male": (
        "Haruto", "Sota", "Yuto", "Ren", "Kaito", "Hiroshi", "Kenji",
        "Takashi", "Daiki", "Ryo", "Satoshi", "Naoki",
    ),
    "japanese_female": (
        "Yui", "Aoi", "Hana", "Sakura", "Rin", "Keiko", "Yuki", "Mai",
        "Akiko", "Emi", "Nana", "Rei",
    ),
    # Korean
    "korean_male": (
        "Minjun", "Seojun", "Doyun", "Jiho", "Junseo", "Minsu", "Jihoon",
        "Sungmin", "Hyunwoo", "Jaewon", "Kyungho", "Donghae",
    ),
    "korean_female": (
        "Seoyeon", "Seoyun", "Jiwoo", "Hana", "Jimin", "Yuna", "Soyeon",
        "Eunji", "Hyejin", "Minji", "Jiyoung", "Sujin",
    ),
    # Vietnamese
    "vietnamese_male": (
        "Minh", "Nam", "Hoang", "Duc", "Tuan", "Khanh", "Bao", "Long",
        "Thang", "Quan", "Hung", "Son",
    ),
    "vietnamese_female": (
        "Linh", "Huong", "Mai", "Thao", "Ngoc", "Trang", "Lan", "Ha",
        "Phuong", "Anh", "Hanh", "Nga",
    ),
    # Anglo (British / American / Australian)
    "anglo_male": (
        "James", "William", "Oliver", "Jack", "Henry", "Thomas", "George",
        "Charlie", "Ethan", "Noah", "Lucas", "Samuel",
    ),
    "anglo_female": (
        "Olivia", "Emma", "Charlotte", "Amelia", "Grace", "Sophie", "Isla",
        "Ava", "Mia", "Ella", "Chloe", "Lily",
    ),
    # Italian
    "italian_male": (
        "Luca", "Marco", "Matteo", "Alessandro", "Giuseppe", "Francesco",
        "Lorenzo", "Andrea", "Davide", "Antonio", "Simone", "Riccardo",
    ),
    "italian_female": (
        "Giulia", "Sofia", "Francesca", "Chiara", "Martina", "Elena",
        "Alessia", "Valentina", "Sara", "Federica", "Giorgia", "Beatrice",
    ),
    # French
    "french_male": (
        "Louis", "Hugo", "Gabriel", "Jules", "Nathan", "Antoine", "Théo",
        "Pierre", "Lucas", "Maxime", "Julien", "Nicolas",
    ),
    "french_female": (
        "Emma", "Léa", "Chloé", "Manon", "Camille", "Sarah", "Louise",
        "Alice", "Juliette", "Clara", "Inès", "Margaux",
    ),
    # German
    "german_male": (
        "Lukas", "Jonas", "Leon", "Finn", "Paul", "Felix", "Maximilian",
        "Elias", "Ben", "Niklas", "Tobias", "Sebastian",
    ),
    "german_female": (
        "Emma", "Mia", "Hannah", "Emilia", "Sophie", "Lena", "Marie",
        "Lea", "Anna", "Laura", "Johanna", "Franziska",
    ),
    # Russian
    "russian_male": (
        "Alexander", "Dmitri", "Ivan", "Sergei", "Mikhail", "Andrei",
        "Nikolai", "Vladimir", "Pavel", "Yuri", "Alexei", "Konstantin",
    ),
    "russian_female": (
        "Anastasia", "Elena", "Olga", "Natalia", "Ekaterina", "Svetlana",
        "Irina", "Maria", "Tatiana", "Yulia", "Ksenia", "Larisa",
    ),
    # Polish
    "polish_male": (
        "Piotr", "Jan", "Krzysztof", "Andrzej", "Tomasz", "Marcin",
        "Michał", "Jakub", "Paweł", "Wojciech", "Grzegorz", "Rafał",
    ),
    "polish_female": (
        "Anna", "Maria", "Katarzyna", "Agnieszka", "Magdalena", "Zofia",
        "Julia", "Ewa", "Aleksandra", "Małgorzata", "Barbara", "Natalia",
    ),
    # Scandinavian
    "scandinavian_male": (
        "Lars", "Erik", "Anders", "Magnus", "Nils", "Johan", "Henrik",
        "Sven", "Bjørn", "Ole", "Gustav", "Fredrik",
    ),
    "scandinavian_female": (
        "Astrid", "Ingrid", "Sofia", "Elin", "Maja", "Freja", "Sanna",
        "Kari", "Linnea", "Sigrid", "Hanne", "Marit",
    ),
    # Greek
    "greek_male": (
        "Georgios", "Dimitrios", "Konstantinos", "Nikos", "Yannis",
        "Christos", "Vasilis", "Panagiotis", "Andreas", "Stavros",
        "Kostas", "Thanasis",
    ),
    "greek_female": (
        "Maria", "Eleni", "Katerina", "Sofia", "Georgia", "Dimitra",
        "Ioanna", "Vasiliki", "Despina", "Angeliki", "Christina", "Fotini",
    ),
    # Turkish
    "turkish_male": (
        "Mehmet", "Mustafa", "Ahmet", "Emre", "Burak", "Yusuf", "Kerem",
        "Deniz", "Baris", "Ozan", "Serkan", "Hakan",
    ),
    "turkish_female": (
        "Elif", "Zeynep", "Merve", "Ayse", "Esra", "Buse", "Selin",
        "Fatma", "Ebru", "Sevgi", "Derya", "Gizem",
    ),
    # Persian (Iranian)
    "persian_male": (
        "Ali", "Reza", "Amir", "Hossein", "Mehdi", "Saeed", "Kaveh",
        "Arash", "Babak", "Omid", "Farhad", "Behrouz",
    ),
    "persian_female": (
        "Fatemeh", "Zahra", "Maryam", "Leila", "Sara", "Nazanin", "Shirin",
        "Parisa", "Yasaman", "Roya", "Mahsa", "Niloofar",
    ),
    # Arab (Levantine)
    "arab_levantine_male": (
        "Mohammed", "Ahmad", "Omar", "Ali", "Khaled", "Yusuf", "Ibrahim",
        "Hassan", "Karim", "Samir", "Bassam", "Tariq",
    ),
    "arab_levantine_female": (
        "Fatima", "Aisha", "Layla", "Noor", "Mariam", "Sara", "Yara",
        "Rania", "Dana", "Hala", "Nadia", "Salma",
    ),
    # Sudanese
    "sudanese_male": (
        "Mohamed", "Osman", "Abdel", "Yousif", "Tariq", "Musa", "Elhadi",
        "Babiker", "Khalid", "Salah", "Idris", "Gamal",
    ),
    "sudanese_female": (
        "Amna", "Zeinab", "Huda", "Mona", "Salma", "Nour", "Sara",
        "Marwa", "Rania", "Somaya", "Duaa", "Isra",
    ),
    # Algerian
    "algerian_male": (
        "Amine", "Yacine", "Karim", "Sofiane", "Bilal", "Rachid", "Nabil",
        "Mourad", "Salim", "Fares", "Hakim", "Riad",
    ),
    "algerian_female": (
        "Amel", "Yasmine", "Nadia", "Sabrina", "Lamia", "Meriem", "Souad",
        "Karima", "Wassila", "Nawel", "Samira", "Hayat",
    ),
    # Moroccan
    "moroccan_male": (
        "Youssef", "Hamza", "Anas", "Mehdi", "Reda", "Ayoub", "Bilal",
        "Othmane", "Zakaria", "Ismail", "Marouane", "Soufiane",
    ),
    "moroccan_female": (
        "Salma", "Imane", "Hajar", "Nada", "Khadija", "Meryem", "Sara",
        "Ghita", "Oumaima", "Chaimae", "Kenza", "Loubna",
    ),
    # Ethiopian
    "ethiopian_male": (
        "Abebe", "Dawit", "Yonas", "Bekele", "Tesfaye", "Getachew",
        "Kebede", "Solomon", "Girma", "Mulugeta", "Fikru", "Habtamu",
    ),
    "ethiopian_female": (
        "Tigist", "Hanna", "Selam", "Meron", "Bethlehem", "Genet",
        "Aster", "Marta", "Rahel", "Sara", "Kalkidan", "Eden",
    ),
    # Rwandan
    "rwandan_male": (
        "Emmanuel", "Eric", "Patrick", "Olivier", "Aime", "Fabrice",
        "Innocent", "Jean", "Thierry", "Gilbert", "Placide", "Theoneste",
    ),
    "rwandan_female": (
        "Chantal", "Claudine", "Divine", "Aline", "Josiane", "Yvette",
        "Solange", "Immaculee", "Vestine", "Consolee", "Beatrice", "Furaha",
    ),
    # Nigerian — Yoruba
    "yoruba_male": (
        "Adewale", "Babatunde", "Femi", "Tunde", "Ayodele", "Segun",
        "Kunle", "Wale", "Bola", "Dele", "Gbenga", "Tayo",
    ),
    "yoruba_female": (
        "Adeola", "Folake", "Yetunde", "Bukola", "Titilayo", "Funke",
        "Simisola", "Ayoola", "Kemi", "Ronke", "Bisi", "Damilola",
    ),
    # Nigerian — Igbo
    "igbo_male": (
        "Chidi", "Emeka", "Obinna", "Ikenna", "Nnamdi", "Uche", "Kelechi",
        "Chukwuma", "Ifeanyi", "Okechukwu", "Chinedu", "Ebuka",
    ),
    "igbo_female": (
        "Ngozi", "Chioma", "Adaeze", "Ifeoma", "Chinwe", "Amara",
        "Uchenna", "Nneka", "Ada", "Ogechi", "Chiamaka", "Ijeoma",
    ),
    # Ghanaian — Akan
    "ghanaian_akan_male": (
        "Kwame", "Kofi", "Yaw", "Kwabena", "Kwaku", "Kojo", "Kwesi",
        "Fiifi", "Ekow", "Kobina", "Yao", "Akwasi",
    ),
    "ghanaian_akan_female": (
        "Ama", "Akosua", "Abena", "Efua", "Adwoa", "Akua", "Yaa",
        "Afia", "Esi", "Adjoa", "Araba", "Aba",
    ),
    # Hispanic (Spanish / Latin American)
    "hispanic_male": (
        "José", "Juan", "Luis", "Carlos", "Miguel", "Javier", "Diego",
        "Antonio", "Manuel", "Francisco", "Alejandro", "Sergio",
    ),
    "hispanic_female": (
        "María", "Carmen", "Ana", "Isabel", "Lucía", "Elena", "Sofía",
        "Laura", "Marta", "Rosa", "Paula", "Andrea",
    ),
    # Brazilian (Portuguese)
    "brazilian_male": (
        "João", "Pedro", "Lucas", "Gabriel", "Rafael", "Bruno", "Thiago",
        "Felipe", "Gustavo", "Rodrigo", "Matheus", "Vinicius",
    ),
    "brazilian_female": (
        "Ana", "Beatriz", "Juliana", "Camila", "Larissa", "Fernanda",
        "Mariana", "Patrícia", "Gabriela", "Amanda", "Bruna", "Carolina",
    ),
    # Filipino
    "filipino_male": (
        "Jose", "Angelo", "Mark", "Paolo", "Miguel", "Carlo", "Emmanuel",
        "Rodel", "Jomar", "Reymark", "Jayson", "Christian",
    ),
    "filipino_female": (
        "Maria", "Angelica", "Kristine", "Jenny", "Grace", "Camille",
        "Divine", "Jasmine", "Mary", "Rowena", "Kimberly", "Aileen",
    ),
    # Indonesian
    "indonesian_male": (
        "Budi", "Adi", "Agus", "Rizki", "Dimas", "Bayu", "Fajar", "Andi",
        "Wahyu", "Eko", "Dedi", "Yusuf",
    ),
    "indonesian_female": (
        "Siti", "Dewi", "Putri", "Ayu", "Rina", "Wulan", "Indah", "Sari",
        "Ratna", "Fitri", "Ani", "Lestari",
    ),
}

# ── surnames, keyed by "<ethnicity>" (unisex) or "<ethnicity>_<gender>" ──────────

LAST_NAMES: dict[str, tuple[str, ...]] = {
    "indian_bengali_hindu": (
        "Das", "Ghosh", "Banerjee", "Chatterjee", "Mukherjee", "Sen",
        "Bose", "Dutta", "Roy", "Chakraborty", "Bhattacharya", "Sarkar",
    ),
    "indian_bengali_muslim": (
        "Rahman", "Chowdhury", "Khan", "Ahmed", "Hossain", "Islam",
        "Uddin", "Alam", "Haque", "Mondal", "Sheikh", "Mia",
    ),
    "indian_north_hindu": (
        "Sharma", "Verma", "Gupta", "Kumar", "Singh", "Yadav",
        "Mishra", "Tiwari", "Agarwal", "Joshi", "Pandey", "Saxena",
    ),
    "indian_north_hindu_female": ("Devi", "Kumari"),
    "indian_punjabi_sikh_male": ("Singh",),
    "indian_punjabi_sikh_female": ("Kaur",),
    "indian_punjabi_sikh": (
        "Gill", "Sidhu", "Dhillon", "Sandhu", "Grewal", "Bajwa",
        "Brar", "Chahal", "Mann", "Sekhon",
    ),
    "indian_tamil": (
        "Subramanian", "Krishnan", "Raman", "Iyer", "Pillai", "Nadar",
        "Rajan", "Murugan", "Sundaram", "Natarajan", "Venkatesan", "Chandran",
    ),
    "indian_gujarati": (
        "Patel", "Shah", "Mehta", "Desai", "Modi", "Joshi", "Trivedi",
        "Amin", "Bhatt", "Vyas", "Parikh", "Gandhi",
    ),
    "chinese": (
        "Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao",
        "Wu", "Zhou", "Xu", "Sun",
    ),
    "japanese": (
        "Sato", "Suzuki", "Takahashi", "Tanaka", "Watanabe", "Ito",
        "Yamamoto", "Nakamura", "Kobayashi", "Kato", "Yoshida", "Yamada",
    ),
    "korean": (
        "Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon",
        "Jang", "Lim", "Han", "Shin",
    ),
    "vietnamese": (
        "Nguyen", "Tran", "Le", "Pham", "Hoang", "Phan", "Vu", "Dang",
        "Bui", "Do", "Ho", "Ngo",
    ),
    "anglo": (
        "Smith", "Jones", "Williams", "Brown", "Taylor", "Davies",
        "Wilson", "Evans", "Thomas", "Roberts", "Johnson", "Walker",
    ),
    "italian": (
        "Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano",
        "Colombo", "Ricci", "Marino", "Greco", "Conti", "Costa",
    ),
    "french": (
        "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard",
        "Petit", "Durand", "Leroy", "Moreau", "Simon", "Laurent",
    ),
    "german": (
        "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer",
        "Wagner", "Becker", "Schulz", "Hoffmann", "Koch", "Bauer",
    ),
    "russian_male": (
        "Ivanov", "Smirnov", "Kuznetsov", "Popov", "Sokolov", "Petrov",
        "Volkov", "Morozov", "Novikov", "Fedorov",
    ),
    "russian_female": (
        "Ivanova", "Smirnova", "Kuznetsova", "Popova", "Sokolova", "Petrova",
        "Volkova", "Morozova", "Novikova", "Fedorova",
    ),
    "polish_male": (
        "Nowak", "Kowalski", "Wiśniewski", "Wójcik", "Kamiński",
        "Lewandowski", "Zieliński", "Szymański", "Woźniak", "Dąbrowski",
    ),
    "polish_female": (
        "Nowak", "Kowalska", "Wiśniewska", "Wójcik", "Kamińska",
        "Lewandowska", "Zielińska", "Szymańska", "Woźniak", "Dąbrowska",
    ),
    "scandinavian": (
        "Andersson", "Johansson", "Karlsson", "Nilsson", "Eriksson",
        "Larsson", "Olsen", "Hansen", "Jensen", "Berg",
    ),
    "greek": (
        "Papadopoulos", "Nikolaou", "Georgiou", "Papadakis", "Vlachos",
        "Antoniou", "Ioannou", "Makris", "Angelopoulos", "Dimitriou",
    ),
    "turkish": (
        "Yilmaz", "Kaya", "Demir", "Sahin", "Celik", "Yildiz", "Aydin",
        "Ozturk", "Arslan", "Dogan", "Kilic", "Aslan",
    ),
    "persian": (
        "Hosseini", "Ahmadi", "Mohammadi", "Rezaei", "Karimi", "Moradi",
        "Jafari", "Rostami", "Tehrani", "Sadeghi", "Kazemi", "Ghorbani",
    ),
    "arab_levantine": (
        "Haddad", "Khoury", "Nassar", "Saleh", "Hariri", "Mansour",
        "Najjar", "Aoun", "Sleiman", "Fares", "Kassem", "Darwish",
    ),
    "sudanese": (
        "Ibrahim", "Abdalla", "Hassan", "Bashir", "Osman", "Ali",
        "Mohamed", "Elamin", "Adam", "Yousif",
    ),
    "algerian": (
        "Benali", "Bouazza", "Haddad", "Belkacem", "Benmoussa", "Cherif",
        "Meziane", "Boudjema", "Bouras", "Amrani",
    ),
    "moroccan": (
        "El Amrani", "Benjelloun", "Alaoui", "Bennani", "Idrissi", "Tazi",
        "Fassi", "Chraibi", "Berrada", "Sabri",
    ),
    "ethiopian": (
        "Tesfaye", "Bekele", "Girma", "Haile", "Assefa", "Tadesse",
        "Mekonnen", "Gebre", "Alemu", "Desta",
    ),
    "rwandan": (
        "Habimana", "Niyonzima", "Nsengiyumva", "Bizimana", "Hakizimana",
        "Mugisha", "Ndayisaba", "Ntawukuriryayo", "Uwimana", "Munyaneza",
    ),
    "yoruba": (
        "Adeyemi", "Ogunleye", "Afolabi", "Balogun", "Adebayo", "Bello",
        "Oladipo", "Ademola", "Ogundana", "Oyelaran",
    ),
    "igbo": (
        "Okafor", "Okonkwo", "Eze", "Nwosu", "Obi", "Okoye", "Nwankwo",
        "Anyanwu", "Onyeka", "Nnadi",
    ),
    "ghanaian_akan": (
        "Mensah", "Owusu", "Boateng", "Osei", "Asante", "Appiah",
        "Agyeman", "Adjei", "Annan", "Ofori",
    ),
    "hispanic": (
        "García", "Rodríguez", "González", "Fernández", "López",
        "Martínez", "Sánchez", "Pérez", "Gómez", "Díaz",
    ),
    "brazilian": (
        "Silva", "Santos", "Oliveira", "Souza", "Costa", "Pereira",
        "Almeida", "Ferreira", "Rodrigues", "Lima",
    ),
    "filipino": (
        "Santos", "Reyes", "Cruz", "Bautista", "Garcia", "Del Rosario",
        "Mendoza", "Aquino", "Ramos", "Villanueva",
    ),
    "indonesian": (
        "Wijaya", "Susanto", "Halim", "Kusuma", "Pratama", "Santoso",
        "Wibowo", "Nugroho", "Hidayat", "Setiawan",
    ),
}

# gender-neutral middle names (kept as a flat pool; middle names are optional
# padding used only when a style needs more distinct combinations)
MIDDLE_NAMES: tuple[str, ...] = (
    "Ann", "Bea", "Blair", "Blake", "Brooke", "Claire", "Cole", "Dale",
    "Dawn", "Dean", "Drew", "Elle", "Eve", "Faith", "Finn", "Grant",
    "Gray", "Gwen", "Hope", "Jade", "Jane", "Jay", "Jean", "Jude", "June",
    "Kai", "Kate", "Kay", "Lane", "Lee", "Mae", "Marie", "Max", "May",
    "Neil", "Noel", "Paige", "Pearl", "Quinn", "Rae", "Ray", "Reed",
    "Reese", "Rose", "Sage", "Shea", "Sky", "Tate", "Tess", "Wren",
)


# ── resolution helpers ──────────────────────────────────────────────────────────

GENDERS = ("male", "female")


def _base_ethnicity(key: str) -> str:
    """Strip a trailing ``_male`` / ``_female`` token from a group key."""
    for g in GENDERS:
        if key.endswith("_" + g):
            return key[: -(len(g) + 1)]
    return key


def _rebuild_indexes() -> None:
    """Recompute the derived aggregates after the base dicts change."""
    global ALL_FIRST, ALL_LAST, ETHNICITIES, FIRST_ETHNICITIES, LAST_ETHNICITIES
    ALL_FIRST = tuple(dict.fromkeys(chain.from_iterable(FIRST_NAMES.values())))
    ALL_LAST = tuple(dict.fromkeys(chain.from_iterable(LAST_NAMES.values())))
    FIRST_ETHNICITIES = tuple(
        dict.fromkeys(_base_ethnicity(k) for k in FIRST_NAMES)
    )
    LAST_ETHNICITIES = tuple(dict.fromkeys(_base_ethnicity(k) for k in LAST_NAMES))
    # ethnicities usable end to end (have both first and last names)
    ETHNICITIES = tuple(e for e in FIRST_ETHNICITIES if e in set(LAST_ETHNICITIES))


ALL_FIRST: tuple[str, ...] = ()
ALL_LAST: tuple[str, ...] = ()
ETHNICITIES: tuple[str, ...] = ()
FIRST_ETHNICITIES: tuple[str, ...] = ()
LAST_ETHNICITIES: tuple[str, ...] = ()
_rebuild_indexes()


def _ethnicity_matches(base: str, ethnicity: str | None) -> bool:
    if ethnicity is None:
        return True
    # exact group ("chinese") or a family prefix ("indian" → indian_bengali_…)
    return base == ethnicity or base.startswith(ethnicity + "_") or base.startswith(ethnicity)


def first_names(gender: str | None = None, ethnicity: str | None = None) -> tuple[str, ...]:
    """First names for a ``gender`` (``male``/``female``/``None`` = both) and an
    optional ``ethnicity`` (exact group or a family prefix like ``indian``).

    Falls back to the full pool when the filters match nothing.
    """
    keys = []
    for key in FIRST_NAMES:
        base = _base_ethnicity(key)
        if gender in GENDERS and not key.endswith("_" + gender):
            continue
        if not _ethnicity_matches(base, ethnicity):
            continue
        keys.append(key)
    pool = tuple(dict.fromkeys(chain.from_iterable(FIRST_NAMES[k] for k in keys)))
    return pool or ALL_FIRST


def last_names(gender: str | None = None, ethnicity: str | None = None) -> tuple[str, ...]:
    """Surnames for a ``gender`` / ``ethnicity``.

    A unisex ``<ethnicity>`` group is always eligible; a gendered
    ``<ethnicity>_male`` / ``<ethnicity>_female`` group is included only for a
    matching (or unspecified) gender, and the opposite gender's group is
    excluded. Falls back to the full pool when nothing matches.
    """
    other = {"male": "female", "female": "male"}.get(gender or "")
    keys = []
    for key in LAST_NAMES:
        base = _base_ethnicity(key)
        if other and key.endswith("_" + other):
            continue
        if not _ethnicity_matches(base, ethnicity):
            continue
        keys.append(key)
    pool = tuple(dict.fromkeys(chain.from_iterable(LAST_NAMES[k] for k in keys)))
    return pool or ALL_LAST


# ── custom library: export / load / install ─────────────────────────────────────

ENV_VAR = "DATA_SAMPLER_NAMES"


def export_library(path: str | Path | None = None) -> str:
    """Serialize the current effective library as an importable module source.

    Hand it to a user to edit; feed the edited file back via :func:`load_library`
    (temporary) or :func:`install_library` (permanent). If ``path`` is given the
    source is also written there.
    """
    def fmt(mapping: dict[str, tuple[str, ...]]) -> str:
        lines = []
        for key, vals in mapping.items():
            inner = ", ".join(repr(v) for v in vals)
            lines.append(f"    {key!r}: ({inner},),")
        return "\n".join(lines)

    src = (
        '"""Custom name library for data-sampler.\n\n'
        "Edit the groups below, then load with data_sampler.load_names_library(path=...)\n"
        "or install permanently with data_sampler.install_names_library(path=...).\n"
        "Group keys are '<ethnicity>_<gender>' for first names and '<ethnicity>'\n"
        "(or '<ethnicity>_<gender>' for gendered surnames) for last names.\n"
        '"""\n\n'
        "FIRST_NAMES = {\n" + fmt(FIRST_NAMES) + "\n}\n\n"
        "LAST_NAMES = {\n" + fmt(LAST_NAMES) + "\n}\n\n"
        "MIDDLE_NAMES = (\n    "
        + ", ".join(repr(v) for v in MIDDLE_NAMES)
        + ",\n)\n"
    )
    if path is not None:
        Path(path).write_text(src, encoding="utf-8")
    return src


def _apply_library(namespace: dict) -> None:
    """Validate a loaded namespace and swap it into the module globals."""
    global FIRST_NAMES, LAST_NAMES, MIDDLE_NAMES
    first = namespace.get("FIRST_NAMES")
    last = namespace.get("LAST_NAMES")
    middle = namespace.get("MIDDLE_NAMES", MIDDLE_NAMES)
    if not isinstance(first, dict) or not isinstance(last, dict):
        raise ValueError(
            "custom names library must define dict FIRST_NAMES and LAST_NAMES"
        )
    if not first or not last:
        raise ValueError("custom names library FIRST_NAMES/LAST_NAMES are empty")
    FIRST_NAMES = {str(k): tuple(v) for k, v in first.items()}
    LAST_NAMES = {str(k): tuple(v) for k, v in last.items()}
    MIDDLE_NAMES = tuple(middle)
    _rebuild_indexes()


def load_library(source: str | None = None, path: str | Path | None = None) -> None:
    """Activate a custom library for the current process (temporary override).

    Pass either module ``source`` text or a ``path`` to a ``.py`` file defining
    ``FIRST_NAMES`` / ``LAST_NAMES`` (and optionally ``MIDDLE_NAMES``). Raises
    ``ValueError`` on an invalid library, leaving the active one unchanged.
    """
    if (source is None) == (path is None):
        raise ValueError("provide exactly one of source or path")
    text = source if source is not None else Path(path).read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(text, str(path or "<names>"), "exec"), ns)  # noqa: S102 - user-supplied library by design
    _apply_library(ns)


def install_library(path: str | Path) -> Path:
    """Permanently install a custom library by overwriting this package's
    ``_names.py`` (takes effect on next import). Needs write permission to the
    install location; prefer :func:`load_library` / ``DATA_SAMPLER_NAMES`` for a
    per-codebase override. Returns the written path."""
    text = Path(path).read_text(encoding="utf-8")
    # validate before clobbering the installed module
    ns: dict = {}
    exec(compile(text, str(path), "exec"), ns)  # noqa: S102
    if not isinstance(ns.get("FIRST_NAMES"), dict) or not isinstance(ns.get("LAST_NAMES"), dict):
        raise ValueError("library to install must define dict FIRST_NAMES and LAST_NAMES")
    target = Path(__file__)
    target.write_text(text, encoding="utf-8")
    _apply_library(ns)  # also activate now
    return target


# per-codebase override: DATA_SAMPLER_NAMES=/path/to/names.py (loaded on import;
# a broken override is ignored so it can never break `import data_sampler`)
_env_override = os.environ.get(ENV_VAR)
if _env_override and Path(_env_override).is_file():
    try:
        load_library(path=_env_override)
    except Exception:  # pragma: no cover - defensive
        pass
