# Research Proposal: Cross-Domain Transfer in Knowledge Graph-Aligned RL for LLM Reasoning

---

## 1. Research Question

**Primary Question:**  
Does training LLMs with knowledge graph path-alignment rewards on diverse KGs enable reasoning transfer to completely different domains, or is the benefit fundamentally domain-specific?

**Secondary Questions:**
- Does multi-KG training improve cross-domain transfer compared to single-KG training?
- Can parameter-efficient methods (LoRA) match full fine-tuning for KG-RL?
- Which reward signals contribute most to transfer?

---

## 2. Motivation & Gap

### Recent Breakthrough
Kansal & Jha (arXiv:2601.15160, January 2026) demonstrated that knowledge graphs can serve as implicit reward models for RL training, enabling compositional multi-hop reasoning. Their SFT → GRPO pipeline with KG path rewards achieved strong results on medical reasoning.

### Critical Gap
Their work—and all existing KG-RL research—evaluates only **in-domain** transfer. **No study has examined whether KG-structured RL training transfers across domains.**

### Supporting Evidence
- arXiv:2507.00432: RL outperforms SFT for cross-domain transfer (SFT causes representation drift; RL preserves latent structure)
- arXiv:2601.14456: Cross-domain transfer is genuinely difficult (models "completely fail on unseen domains")

---

## 3. Proposed Method: KG-Align-RL

### Pipeline
```
Diverse KGs (Freebase, Wikidata, ConceptNet)
                ↓
      Path Extraction (1-3 hop)
                ↓
       Question Generation
                ↓
      Base LLM (Qwen2.5-7B)
                ↓
      GRPO with KG Rewards + LoRA
                ↓
    Evaluation: In-domain → Transfer → Zero-shot
```

### Differentiators from arXiv:2601.15160

| Aspect | Prior Work | Our Work |
|--------|------------|----------|
| Transfer study | None | **Core contribution** |
| KG diversity | Single (medical) | Multiple (3 KGs) |
| Held-out eval | Same domain | **KinshipQA** |
| Efficiency | Full fine-tuning | **LoRA** |

---

## 4. Evaluation Strategy

| Benchmark | Purpose |
|-----------|---------|
| Training KG subset | In-domain baseline |
| HotpotQA | Near transfer |
| MuSiQue | Multi-hop transfer |
| **KinshipQA** | **Zero-shot held-out** |

---

## 5. Timeline & Compute

**Resource:** ~10,000 Isambard GPU hours

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 1 | Months 1-2 | Implement pipeline, replicate baseline |
| Phase 2 | Months 3-5 | Core transfer experiments |
| Phase 3 | Months 6-8 | Ablations, analysis, writing |

---

## 6. Expected Contributions

1. First systematic cross-domain study of KG-RL for LLM reasoning
2. Diverse KG training analysis (single vs. multi-KG)
3. Novel step-wise rewards beyond path alignment
4. Parameter-efficient KG-RL demonstration
5. KinshipQA as contamination-free transfer benchmark

---

## 7. Thesis Fit

| Chapter | Approach | Contribution |
|---------|----------|--------------|
| Ch1: KGEIR | Inference-time KG | Plug-and-play framework |
| Ch2: KinshipQA | Better evaluation | Contamination-proof benchmark |
| **Ch3: KG-Align-RL** | **Training-time KG** | **Transfer study** |

**Arc:** From external KG guidance → to internalized KG reasoning

---

*Proposal prepared: January 2026*
