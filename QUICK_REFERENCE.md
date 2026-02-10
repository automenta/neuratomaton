# Quick Reference Index
## Bio-ANA Project Navigation Guide

**Last Updated**: 2026-02-10  
**Status**: Ready to Execute

---

## Quick Start Commands

### Run WikiText-2 Validation (6 hours)
```bash
python run_wikitext_validation.py \
  --variant small \
  --vocab-size 10000 \
  --seq-len 128 \
  --batch-size 16 \
  --epochs 5 \
  --output results/wikitext2_small
```

### Run Synthetic Validation (30 minutes)
```bash
python validate_stage0.py
```

### Profile Performance (5 minutes)
```bash
python detailed_profile.py
```

---

## Document Index

### Primary Documentation

| Document | Purpose | Status |
|----------|---------|--------|
| **RESEARCH_ROADMAP.md** | Complete experimental plan | ✅ READY |
| **PLAN.md** | Project status & milestones | ✅ UPDATED (v4.0) |
| **FINAL_STATUS.md** | Session summary & achievements | ✅ COMPLETE |
| **REVISED_NEXT_STEPS.md** | Strategic pivot decisions | ✅ DOCUMENTED |

### Supporting Documentation

| Document | Purpose | Section |
|----------|---------|---------|
| **PHASE3_PROFILING_SUMMARY.md** | Profiling findings | Optimization results |
| **SESSION_SUMMARY.md** | Session recap | What was accomplished |

---

## File Reference

### Core Implementation

```
ana/
├── bio_ana/                    # Main architecture
│   ├── config.py              # Model configurations
│   ├── tracks.py              # Multi-track dynamics
│   ├── hololink.py            # HoloLink memory
│   └── model.py               # BioANAModel
├── bio_training/              # Training infrastructure
│   ├── curriculum.py          # Data loaders
│   ├── trainer.py             # Training logic
│   └── scheduler.py           # Scheduling
└── eqprop/                    # EqProp library (cloned)
```

### Scripts & Tools

| Script | Purpose | Time |
|--------|---------|------|
| `run_wikitext_validation.py` | WikiText-2/103 training | 6-20 hours |
| `validate_stage0.py` | AR task validation | 30 minutes |
| `run_curriculum.py` | Full curriculum training | 1-2 hours |
| `detailed_profile.py` | Performance profiling | 5 minutes |
| `quick_profile.py` | Quick timing test | 1 minute |

### Results Directory

```
results/
├── profiling/                  # Performance data
│   └── phase3_optimization_findings.json
├── phase3/                     # Phase 3 results
├── wikitext2_validation/       # WikiText results
└── experiments/                # Experiment outputs
```

---

## Success Criteria Quick Reference

### Tier 1: Proof (Already Achieved ✅)
- [x] AR accuracy: 98% → **100%**
- [x] Optimization: 2x → **5.31x**
- [x] Energy convergence: < 50 iters → **25 iters**

### Tier 2: Validation (Next Target 🎯)
- [ ] WikiText-2 PPL: **< 35** (expected 28-32)
- [ ] Training speed: **> 500 tok/sec**
- [ ] Memory: **< 2GB**

### Tier 3: Publication (Final Goal 📄)
- [ ] WikiText-103 PPL: **< 35**
- [ ] Efficiency vs Transformer: **> 2x**
- [ ] INT8 loss: **< 2%**

---

## Decision Matrix

### DP1: After WikiText-2 (Day 1)

| PPL Range | Decision | Next Step |
|------------|----------|-----------|
| < 30 | **YES** | Full evaluation |
| 30-35 | **YES** | Full evaluation |
| 35-40 | **MAYBE** | Debug (3 hours) |
| 40-50 | **MAYBE** | Review architecture |
| > 50 | **NO** | Analyze failure |

### DP2: After Full Evaluation (Week 2)

| Criteria | Required | Check |
|----------|----------|-------|
| PPL competitive | < 35 | ✅/❌ |
| Efficiency > 2x | Yes/No | ✅/❌ |
| Novel contribution | Yes/No | ✅/❌ |
| Statistical significance | p < 0.05 | ✅/❌ |

---

## Timeline Summary

```
Week 1: WikiText-2 Validation
  Day 1-2: Training (6 hours)
  Day 2-3: Analysis + Decision
  Day 3-5: Continue if success

Week 2: Full Evaluation
  Day 1: Baselines (8 hours)
  Day 2-3: Scale-up (20 hours)
  Day 4: Synthetic tasks (3 hours)
  Day 5: Deployment (5 hours)

Week 3: Paper Writing
  Day 1-2: Analysis & figures
  Day 3-4: Draft manuscript
  Day 5: Internal review

Week 4: Submission
  Day 1: Final revisions
  Day 2: ArXiv preprint
  Day 3-4: Conference submission
```

---

## Resource Summary

| Resource | Amount | Cost |
|----------|--------|------|
| GPU hours | 41 | $123 |
| Human hours | 100 | Internal |
| Storage | 100GB | $50 |
| **Total** | - | **~$200** |

---

## Quick Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Loss explodes | Reduce LR to 5e-4 |
| PPL not decreasing | Increase epochs to 10 |
| Out of memory | Reduce batch size to 8 |
| Too slow | Reduce relaxation to 5 iters |
| NaN errors | Add gradient clipping |

### Performance Issues

| Problem | Fix |
|---------|-----|
| Training too slow | Check adaptive relaxation enabled |
| Memory too high | Verify spectral norm on W_rec |
| Not converging | Increase relaxation iterations |
| Unstable training | Add warmup schedule |

---

## Contact & Support

For reproduction issues:
1. Check `RESEARCH_ROADMAP.md` Appendix
2. Review logs in `results/`
3. Include: config + hardware + full output

---

## Status Overview

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: EqProp | ✅ Complete | 100% |
| Phase 2: Architecture | ✅ Complete | 100% |
| Phase 3: Training | ✅ Complete | 100% |
| Phase 3.5: WikiText | 🚀 Ready | 0% (next step) |
| Phase 4: Full Eval | 📅 Planned | 0% |
| Phase 5: Publication | 📅 Planned | 0% |

**Overall**: 60% complete, ready for data-intensive phase

---

## Next Action

**Status**: ✅ READY  
**Command**: `python run_wikitext_validation.py`  
**Expected**: 6 hours → PPL 28-35  
**Decision**: After results → Go/No-Go

---

## Key Files to Review Before Starting

1. **RESEARCH_ROADMAP.md** - Full experimental plan
2. **PLAN.md** - Project status and milestones
3. **FINAL_STATUS.md** - What we've achieved
4. **ana/bio_training/trainer.py** - Training implementation

---

**Ready to execute**: Run `python run_wikitext_validation.py` to begin Phase 3.5 validation.
