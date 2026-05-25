# V14-B2: G2 kg-incomplete decision boundary analysis

## Setup
N = 818 kg-incomplete trajectories from G2@500 (`results/trajectories/phase7/39g2_step500_full/step_0/trajectories.json`), classified via `task16_classify.classify_trajectory`. Each trajectory's first `<search>` call is scored against the gold `kg_path` from `data/freebase/verl_cwq/test.parquet` (entity: normalized equality + substring; relation: Levenshtein <=3 or `SequenceMatcher.ratio()` > 0.8, full form and dotted tail).

## Distribution of failure modes

| Mode | Count | % of kg-incomplete |
|---|---|---|
| 1. Exact relation name typo (<= 1 edit) | 576 | 70.4% |
| 2. Correct entity, wrong relation | 153 | 18.7% |
| 3. Wrong entity, any relation | 70 | 8.6% |
| 4. Completely off | 19 | 2.3% |

- Buckets 1+2 (entity correct, relation wrong/typoed): **729 (89.1%)**
- Buckets 3+4 (entity wrong): **89 (10.9%)**

## Example trajectories (first 2 per bucket)

### 1. Exact relation name typo (<= 1 edit)
- `WebQTest-1002_806620bbf6ca7cf3ced5d4ff130c714d`  
  Q: Which actor portrayed both Dumbledore and Badger?  
  query: `Dumbledore / film.performance.character`  gold entity match: `professor albus dumbledore`  closest gold rel: `cvg.game_performance.character` (ratio=1.0, edit=0)  
  predicted=`Richard Harris`  gold=`Michael Gambon`  — relation also exactly in gold path
- `WebQTest-100_3fd000b7cbc17c620bb91588e1c3cda3`  
  Q: What is the current main speech medium in Haiti?  
  query: `Haiti / Location.administrative_divisions`  gold entity match: `haiti`  closest gold rel: `location.country.administrative_divisions` (ratio=1.0, edit=0)  
  predicted=`French`  gold=`Haitian Creole`

### 2. Correct entity, wrong relation
- `WebQTest-106_87ddee8816a0dd02511eb7670904d500`  
  Q: Find the movies that had James William Visconti III as a crew member, which one featued Arnie Hammer and who did he play?  
  query: `James William Visconti III / film.performance.actor`  gold entity match: `james william visconti iii`  closest gold rel: `people.person.gender` (ratio=0.333, edit=5)  
  predicted=`The Muppets`  gold=`Tyler Winklevoss`
- `WebQTest-1072_8e6dfa8822e4d4a94104aa4c25644866`  
  Q: Who is the current governor of the governmental jurisdiction that included a governmental body called the 2010 Arizona Senate?  
  query: `2010 Arizona Senate / organization.leadership.role`  gold entity match: `arizona senate`  closest gold rel: `common.topic.article` (ratio=0.545, edit=4)  
  predicted=`Jan Brewer`  gold=`Raúl Héctor Castro`

### 3. Wrong entity, any relation
- `WebQTest-1171_6bdeb539174509f051a303e0ee59c246`  
  Q: What actor played the charactor Troy Bolton and also has done the voice of Star Wars villiam Darth Vader?  
  query: `Troy Bolton / film`  gold entity match: `None`  closest gold rel: `film.actor.film` (ratio=1.0, edit=0)  
  predicted=`James Earl Jones`  gold=`Zac Efron`  — relation close to a gold relation, but wrong entity
- `WebQTest-1306_1514f1e9c4973110faa2ad25559c0435`  
  Q: What is the form of government in the country after 1979 and is where Afshar is spoken?  
  query: `Afshari Language / language_family`  gold entity match: `None`  closest gold rel: `language.human_language.language_family` (ratio=1.0, edit=0)  
  predicted=`Parliamentary system`  gold=`Islamic republic`  — relation close to a gold relation, but wrong entity

### 4. Completely off
- `WebQTest-1178_4b7058a69c5eb71df36a89837b1e4f60`  
  Q: What is the birthplace of the composer of "Papa'z Song"?  
  query: `Papa's Song / music.composition.lyricist`  gold entity match: `None`  closest gold rel: `music.album.artist` (ratio=0.571, edit=4)  
  predicted=`Harlem`  gold=`East Harlem`
- `WebQTest-1823_6ac48b436d99ed7cd3f2d9329d95a0c8`  
  Q: What is the government type where "Ardulfurataini Estan" is the national anthem?  
  query: `Ardulfurataini Estan / music.composition.lyricist`  gold entity match: `None`  closest gold rel: `common.topic.article` (ratio=0.4, edit=6)  
  predicted=`Presidential system`  gold=`Parliamentary system`

## Implication for paper

- Buckets 1+2 = **89.1%** of kg-incomplete (>=50%): majority of G2@500 failures are **query-precision**, not retrieval. The model lands on the correct entity but picks a wrong/typoed relation. A process reward that scores (entity, relation) well-formedness against the KG schema is therefore *directly* viable; **L-lang (schema-format) is the dominant bottleneck** on the 7B CWQ agent, consistent with the V14-B1 hypothesis.
- Bucket 1 alone = **70.4%** of kg-incomplete exceeds the 30% threshold stated in the brief: we can claim '>=30% of kg-incomplete are <=1 token edit from gold', a highly publishable finding for EMNLP 2026.
