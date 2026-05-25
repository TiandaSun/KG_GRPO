# V14-B1: G2@500 failure-mode distribution

## Setup
N=2119 non-EM trajectories from G2@500 (CWQ test, 3,531 total). Classified via the 4 signal-theoretic decision rules below; ties broken in order L-prior > L-sig > L-comp > L-lang.

## Distribution
| Mode | Count | % of non-EM | Paper implication |
|---|---|---|---|
| L-prior | 284 | 13.4% | Model skipped or mis-aimed tool use (no question entity in any query); relies on parametric prior. |
| L-sig | 1170 | 55.21% | Tool channel returned empty/error responses >=50% of the time; answer could not be grounded in retrieved evidence. |
| L-comp | 165 | 7.79% | Chain broke across turns: a later search entity was not in any prior tool response nor the question (state-carrying failure). |
| L-lang | 500 | 23.6% | Tool responses were non-empty but (entity, relation) did not match the gold kg_path triples — schema/string-format mismatch or wrong relation. |

## Decision rules (for reproducibility)
- L-prior: num_tool_calls==0 OR no queried entity appears in the question text (parametric / irrelevant queries).
- L-sig: >=1 tool call AND >=50% of tool responses are empty `[]`/`{}` or error.
- L-comp: >=2 tool calls AND a later search entity is neither in the question nor in any prior tool response (broken entity chain).
- L-lang: default — non-empty responses but (entity, relation) does not match any gold kg_path triple (schema/string-format mismatch).

## Exemplars
### L-prior
- **WebQTrn-60_68f0d0ad309d64a4af858a5ef4fb5713** (hops=1): Q: _In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?_
  - gold: `Mozambique` | predicted: `Brazil`
  - trajectory: `<think>I need to find information about Portuguese Language. Let me start by checking its relations in the knowledge graph.</think> <search>get_tail_entities(Portuguese Language, language.writing_system)</search> <tool_r...`
- **WebQTrn-710_c264a6d11d7956741926d417b94327e2** (hops=0): Q: _When was the last time a team led by Larr Baer win the World Series?_
  - gold: `2014 World Series` | predicted: `1988 World Series`
  - trajectory: `<think>I need to find information about Larr Kim. Let me start by checking its relations in the knowledge graph.</think> <search>get_tail_entities(Larry Baer, sports.sports_team_coach.team)</search> <tool_response>[]</to...`

### L-sig
- **WebQTrn-241_dfb6c97ac9bf2f0ac07f27dd80f9edc2** (hops=1): Q: _What country bordering France contains an airport that serves Nijmegen?_
  - gold: `Germany` | predicted: `Netherlands`
  - trajectory: `<think>I need to find information about France. Let me start by checking its relations in the knowledge graph.</think> <search>get_tail_entities(France, Location.administrative_divisions)</search> <tool_response>[]</tool...`
- **WebQTrn-1077_f4a9e5f1e0dcfb82cbadf4771eda7bb5** (hops=0): Q: _The national anthem Afghan National Anthem is from the country which practices what religions?_
  - gold: `Shia Islam` | predicted: `Islam`
  - trajectory: `<think>I need to find information about Afghan National Anthem. Let me start by checking its relations in the knowledge graph.</think> <search>get_tail_entities(Afghan National Anthem, music_composer)</search> <tool_resp...`

### L-comp
- **WebQTest-1686_29e74083744b3631541b29b4094fb273** (hops=2): Q: _Who was the governor of Arizona in 2009 that held his governmental position before 1998?_
  - gold: `Janet Napolitano` | predicted: `Jan Brewer`
  - trajectory: `<think>I need to find information about Governor. Let me start by checking its relations in the knowledge graph.</think> <search>get_tail_entities(Governor, common.topic.notable_types)</search> <tool_response>["Film char...`
- **WebQTest-998_f693ae2cb9aa6b8ffdf0f103a6777b62** (hops=2): Q: _What is the most recent event in which the sports team owned by James L. Dolan won the championship?_
  - gold: `1973 NBA Finals` | predicted: `2014 NBA Finals`
  - trajectory: `<think>I need to find information about James L. Dolan. Let me start by checking its relations in the knowledge graph.</think> <search>get_tail_entities(James L. Dolan, organization.organization_founder.organizations_fou...`

### L-lang
- **WebQTest-832_c334509bb5e02cacae1ba2e80c176499** (hops=2): Q: _Lou Seal is the mascot for the team that last won the World Series when?_
  - gold: `2014 World Series` | predicted: `2010 World Series`
  - trajectory: `<think>I need to find information about Lou Seal. Let me start by checking its relations in the knowledge graph.</think> <search>get_tail_entities(Lou Seal, sports.mascot.team)</search> <tool_response>["San Francisco Gia...`
- **WebQTrn-1259_1997cb4922db71983be26e6a509950f4** (hops=0): Q: _Where did the "Country Nation World Tour" concert artist go to college?_
  - gold: `Belmont University` | predicted: `University of Tennessee`
  - trajectory: `<think>I need to find information about Country Nation World Tour. Let me start by checking its relations in the knowledge graph.</think> <search>get_tail_entities(Country Nation World Tour, music.concert_tour.artist)</s...`

## Headline finding
The dominant failure mode is **L-sig** (55.21% of non-EM), followed by **L-lang** (23.6%). The model can identify the right entity and relation shape but the KG channel returns no evidence — G2 is retrieval-limited, not schema-limited.
