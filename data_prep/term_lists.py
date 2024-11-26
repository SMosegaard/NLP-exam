from itertools import chain
import pandas as pd

female_names = pd.read_csv('/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data/female_names.csv')
male_names = pd.read_csv('/work/SofieNørboMosegaard#5741/NLP/NLP-exam/data/male_names.csv')

# Mask names
# Define name sets
female_names = female_names['Name'].tolist()
male_names = male_names['Name'].tolist()

# Female or male words, where there is no oppsite
female_specific_terms = ['blondine', 'groupie', 'gravid', 'luder', 'dulle', 'jomfruhinde',
                        'moderkage', 'abort', 'babe', 'nymfe', 'bagerjomfru', 'bh', 'bimbo',
                        'skabslebbe', 'skabslesbisk',
                        ]

male_specific_terms = ['playboy', 'ungkarl', 'Spider-Man', 'macho', 'machoprægede', 'metroseksuel',
                        'bejler', 'blowjob',
                        ]


# only listing the "base-word" of the terms (so "kvinde" and not "kvinderne")
term_dict = {

    # female/male, family, relatives, etc. 
    'kvinde': 'mand',
    'pige': 'dreng',
    'søster': 'bror',
    'søstre': 'brødre',
    'datter': 'søn',
    'døtre': 'sønner',
    'mor': 'far', 
    'mødre': 'fædre',
    'moder': 'fader',
    'dame': 'mand',
    'damer': 'mænd',
    'drengepige': 'pigedreng',
    'kone': 'husbond',
    'enke': 'enkemand',
    'enker': 'enkemænd',
    'frue': 'herre',
    'hustru': 'husbond',
    'tante': 'onkel',
    'moster': 'morbror',
    'mostre': 'morbrødre',
    'svigerinde': 'svoger',
    'svigerinder': 'svogre',
    'veninde': 'ven',

    # English female/male terms
    'woman': 'man',
    'women': 'men',
    'mamma': 'pappa',
    'girl': 'boy',
    'gal': 'guy',
    'gals': 'guys',
    'mrs': 'mr',

    # Descriptive 
    'kvindelig': 'mandlig',
    'kvindelig': 'mandig',
    'moderlig': 'faderlig',
    'piget': 'drenget',
    'pigede': 'drengede', 
    'feminim': 'maskulin',
    'søsterlig': 'broderlig',
    'lesbiske': 'bøssede', 

    # Roles and occupations
    'elskerinde': 'elsker',
    'værtinde': 'vært',
    'journalistinde': 'journalist',
    'heltinde': 'helt',
    'fjendinde': 'fjende', 
    'ballerina': 'ballerino',
    'bestyrerinde': 'bestyrer',
    'dronning': 'konge',
    'lesbisk': 'bøsse', 
    'danserinde': 'danser',
    'forstanderinde': 'forstander',
    'gentlewoman': 'gentleman',
    'gentlewomen': 'gentlemen',
    'skuespillerinde': 'skuespiller',
    'heks': 'troldmand',
    'hekse': 'troldmænd',
    'lærerinde': 'lærer',
    'nonne': 'munk',
    'prinsesse': 'prins',
    'troldkvinde': 'troldmand',
    'troldkvinder': 'troldmænd',
    'brud': 'brudgom',

    # nouns
    'yoni': 'fallos',
    'kusse': 'pik',
    'bryster': 'brystkasse',
    'bikini': 'badeshorts',
    'badedragt': 'badeshorts',
    'girlband': 'boyband', 

}

pron_dict = {
     'hun': 'han',
     'hende': 'ham',
     'hendes': 'hans'
}


weat_dict = {
    # still needs to be defined 
    "kvinde" : "mand",
    "søster" :  "bror",
    "mor" : "far",
    "hende" : "ham",
}

    

##################################################


# Combine all terms (female + male + pronouns)
all_terms = list(term_dict.keys()) + list(term_dict.values()) + list(pron_dict.keys()) + list(pron_dict.values())

# Extract all pronouns
all_prons = list(pron_dict.keys()) + list(pron_dict.values())

# Combine term_dict and pron_dict into one dictionary
term_dict = dict(chain(term_dict.items(), pron_dict.items()))

# Gender-specific terms dictionaries
    # (Female-specific names/terms should be removed in the male version)
    # (Male-specific names/terms should be removed in the female version)
    # (as female terms are first in the dict = term_dict.keys() = k)
    # (as male terms are second in the dict = term_dict.values() = v)

female_names_dict = {name: "" for name in female_names}  # Female names should be replaced with ""
male_names_dict = {name: "" for name in male_names}

female_specific_terms_dict = {term: "" for term in female_specific_terms}  # Female terms should be replaced with ""
male_specific_terms_dict = {term: "" for term in male_specific_terms}

# Female to male: remove female-specific terms in male version
terms_f2m = {**term_dict, **female_names_dict, **female_specific_terms_dict}

# Male to female: remove male-specific terms in female version
terms_m2f = {**dict((k, v) for v, k in term_dict.items()), **male_names_dict, **male_specific_terms_dict}


# Define the "all terms" incl female/male_specific_dict
all_terms = set(list(terms_m2f.keys()) + list(terms_f2m.keys()))

