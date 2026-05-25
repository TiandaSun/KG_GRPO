# V16-Q8-REDO Phase 1 sanity report (REGEN approach)

- Source parquet: `data/freebase/verl_cwq_q8_redo/train.parquet`
- Generated SFT JSONL: `data/freebase/sft_trajectories_q8_redo.jsonl`
- Trajectory count: 5000

## Check summary

| # | Check | Result | Failures |
|---|-------|--------|----------|
| 1 | system prompt byte-identical to canonical | PASS | 0 / 5000 |
| 2 | entity:// in user/asst/tool content of 5 samples | PASS | 0 / 5 |
| 3 | no nested entity://...entity:// corruption | PASS | 0 |
| 4 | no entity:// in system prompt | PASS | 0 |

## Check 2: 5 random trajectories (question + first assistant turn + first tool response)

### Trajectory 912  (URIs in non-system content: True)

**Question:**

```
What college has 5835 postgraduates and includes entity://jay_cutler__96dccc52 in their list of alumni?
```

**First assistant turn:**

```
<think>I need to find information about entity://jay_cutler__96dccc52. Let me start by checking its relations in the knowledge graph.</think>
<search>get_tail_relations(entity://jay_cutler__96dccc52)</search>
```

**First tool response (user-role w/ <tool_response>):**

```
<tool_response>["american_football.football_player.rushing", "medicine.notable_person_with_medical_condition.condition", "people.person.spouse_s"]</tool_response>
```

### Trajectory 204  (URIs in non-system content: True)

**Question:**

```
What division is team owner entity://mike_brown__456627ca's sports team?
```

**First assistant turn:**

```
<think>I need to find information about entity://mike_brown__456627ca. Let me start by checking its relations in the knowledge graph.</think>
<search>get_tail_relations(entity://mid__c4db5cb0)</search>
```

**First tool response (user-role w/ <tool_response>):**

```
<tool_response>["sports.pro_sports_played.athlete"]</tool_response>
```

### Trajectory 2253  (URIs in non-system content: True)

**Question:**

```
Which city of residence for entity://barack_obama__21586586 has a neighborhood named entity://brentwood__57caeabd?
```

**First assistant turn:**

```
<think>I need to find information about entity://barack_obama__21586586. Let me start by checking its relations in the knowledge graph.</think>
<search>get_tail_relations(entity://us_senator_barack_obama_podcast__3c2e3dd6)</search>
```

**First tool response (user-role w/ <tool_response>):**

```
<tool_response>["broadcast.content.producer"]</tool_response>
```

### Trajectory 2006  (URIs in non-system content: True)

**Question:**

```
What is the birthplace of the subject in the film titled entity://the_diary_of_anne_frank__becb4842?
```

**First assistant turn:**

```
<think>I need to find information about entity://the_diary_of_anne_frank__becb4842. Let me start by checking its relations in the knowledge graph.</think>
<search>get_tail_relations(entity://the_diary_of_anne_frank__becb4842)</search>
```

**First tool response (user-role w/ <tool_response>):**

```
<tool_response>["film.film.language", "film.film.subjects", "film.film.country"]</tool_response>
```

### Trajectory 1828  (URIs in non-system content: True)

**Question:**

```
Who did the nominee for entity://miss_march__0179ba80 date in 2012?
```

**First assistant turn:**

```
<think>I need to find information about entity://miss_march__0179ba80. Let me start by checking its relations in the knowledge graph.</think>
<search>get_tail_relations(entity://miss_march__0179ba80)</search>
```

**First tool response (user-role w/ <tool_response>):**

```
<tool_response>["film.film.starring", "film.film.starring", "common.topic.notable_types"]</tool_response>
```

## Overall verdict

**ALL CHECKS PASS**
