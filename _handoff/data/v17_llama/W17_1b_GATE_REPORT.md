# v17 W17.1 format-emission gate report

- Trajectories analysed: **100**
- `<answer>` open tag emitted: 0/100 (0.0%)
- `</answer>` close tag emitted: 0/100 (0.0%)
- Both tags present (format-valid): **0/100 (0.0%)**
- Both tags + non-empty answer: 0/100 (0.0%)
- Mean tool calls / question: **5.00**
- Strict EM (sanity, not load-bearing): 0.0000
- Token-F1 (sanity): 0.0226

## Gate decision

- format_valid >= 90%: FAIL (0.0%)
- tools_per_q > 0.5: PASS (5.00)

**Verdict: ABORT**

Per v17 spec W17.1: if format-valid < 90%, STOP — do NOT proceed to W17.2. Log findings; rebuttal-only ammunition.

## Sample failure modes (first 3)

```
sample_id=WebQTest-832_c334509bb5e02cacae1ba2e80c176499 | gold=2014 World Series | open=False close=False
  ...'notable_for\' to continue reasoning.</think>\n<search>get_tail_entities(Lou Seal, common.topic.notable_for)</search>user\n\n<tool_response>["g.1259jyj2_"]</tool_response>\n<tool_response>["g.125bn82w_"]</tool_response>\n<think>Now I know about g.125bn82w_. Let me follow the relation \'common.topic.notable_types\' to continue reasoning.</think>\n<search>get_tail_entities(Lou Seal, common.topic.notable_types)</search>user\n\n<tool_response>["Mascot"]</tool_response>\n<tool_response>["Mascot"]</tool_response>\n'
```

```
sample_id=WebQTrn-1259_1997cb4922db71983be26e6a509950f4 | gold=Belmont University | open=False close=False
  ...'ntry Nation World Tour, music.concert_tour.artist)</search>\n<tool_response>["Brad Paisley"]</tool_response>\n<think>Now I know about Brad Paisley. Let me follow the relation\'music.artist.concert_tours\' to continue reasoning.</think>\n<search>get_tail_entities(Brad Paisley, music.artist.concert_tours)</search>user\n\n<tool_response>["Brad Paisley 2009 Concert Tour"]</tool_response>\n<tool_response>["Beat This Summer Tour", "Brad Paisley 2009 Concert Tour", "Country Nation World Tour"]</tool_response>\n'
```

```
sample_id=WebQTest-1384_744a496b907e407b16bc5d7c197dc3f0 | gold=Judaism | open=False close=False
  ...'2qmp2. Let me follow the relation \'common.topic.image\' to continue reasoning.</think>\n<search>get_tail_entities(Ovadia Yosef, common.topic.image)</search>\n<tool_response>["Ovadya Yosef"]</tool_response>\n<think>Now I know about Ovadya Yosef. Let me follow the relation \'common.topic.notable_for\' to continue reasoning.</think>\n<search>get_tail_entities(Ovadia Yosef, common.topic.notable_for)</search>user\n\n<tool_response>["g.1259j5j2b"]</tool_response>\n<tool_response>["g.125f2ttc4"]</tool_response>\n'
```

