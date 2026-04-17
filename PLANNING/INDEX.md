# IrisFlow Rust + ML Rewrite: Complete Planning Index

This index is your roadmap for the 12-16 week rewrite project combining Rust systems programming with machine learning.

---

## Start Here

**New to this project?** Read these in order:

1. **00_GETTING_STARTED.md** — Week 1 checklist and immediate next steps
2. **RUST_ML_REWRITE_PLAN.md** — Full 16-week master plan with phase breakdown
3. Choose your focus:
   - **01_DATA_COLLECTION_GUIDE.md** — Detailed ML data collection methodology
   - **02_RUST_ARCHITECTURE.md** — System architecture and module design
   - **03_LEARNING_OBJECTIVES.md** — Skills, learning path, and progress tracking

---

## Document Purposes

### 00_GETTING_STARTED.md
**Quick start for Week 1 (Days 1-7)**
- Pre-project hardware verification
- Directory structure setup
- Day-by-day checklist for data collection
- Quick decision tree for common issues
- Expected artifacts by end of week

**Read this if:** You're starting today and need immediate next steps.

### RUST_ML_REWRITE_PLAN.md
**Complete master plan (16 weeks, ~300-400 hours)**
- 3-phase breakdown: ML Data (Weeks 1-4), Rust System (Weeks 5-12), Polish (Weeks 13-16)
- Detailed deliverables for each sub-phase (2.1, 2.2, 2.3, etc.)
- Week-by-week timeline with checkpoints
- Risk mitigation strategies
- Success criteria (technical, portfolio, learning)

**Read this if:** You want the full picture before committing.

### 01_DATA_COLLECTION_GUIDE.md
**Detailed ML methodology (Phase 1.1-1.2)**
- Part 1: Gaze dataset collection (9-point grid methodology)
- Part 2: Gesture dataset collection (gesture-by-gesture recording)
- Python scripts for automated collection and validation
- Data quality checks and visualization
- Tips for high-quality data (lighting, camera angle, rest breaks)
- Storage format specification (JSON for gaze, pickle for gestures)

**Read this if:** You're ready to collect data or need to understand the ML pipeline.

### 02_RUST_ARCHITECTURE.md
**System design and implementation strategy (Phase 2)**
- System overview diagram and data flow
- Complete module structure (src/ layout)
- Key data structures (Frame, GazeResult, GestureResult, etc.)
- Core algorithms: One-Euro filter, temporal gesture filtering, ring buffer
- Real-time pipeline execution flow with latency targets
- Memory management and unsafe Rust usage
- Platform-specific implementations (macOS vs. Linux)
- Performance targets and optimization checklist

**Read this if:** You're ready to start Rust development or want to understand the architecture.

### 03_LEARNING_OBJECTIVES.md
**Educational roadmap and skill development (Weeks 1-16)**
- Rust systems programming: 6 modules (ownership, FFI, async, profiling, low-level APIs, deployment)
- Machine learning: 5 phases (dataset, regression, sequences, export, systems)
- Checkpoint exercises for each module (hands-on learning)
- Comprehensive skill matrix (beginner → expert)
- Weekly checkpoint template for self-assessment
- Portfolio artifacts you'll create
- Resource compilation (books, tutorials, documentation)
- Success definition for the full project

**Read this if:** You want to maximize learning or understand skill progression.

---

## Quick Navigation by Role

### "I want to code right now"
→ **00_GETTING_STARTED.md** (Week 1 tasks)

### "I want the full plan before I start"
→ **RUST_ML_REWRITE_PLAN.md** (read executive summary + timeline)

### "I'm a machine learning person"
→ **01_DATA_COLLECTION_GUIDE.md** + **03_LEARNING_OBJECTIVES.md** (ML section)

### "I'm a systems programmer"
→ **02_RUST_ARCHITECTURE.md** + **03_LEARNING_OBJECTIVES.md** (Rust section)

### "I want to learn both Rust and ML"
→ **03_LEARNING_OBJECTIVES.md** (complete learning path)

### "I need to understand the architecture"
→ **02_RUST_ARCHITECTURE.md** (system design + implementation)

---

## Timeline Summary

| Phase | Weeks | Focus | Deliverables |
|-------|-------|-------|--------------|
| **Phase 1:** ML Foundation | 1-4 | Data collection + model training | 2 trained ONNX models |
| **Phase 2:** Rust System | 5-12 | Real-time system implementation | Standalone Rust binary |
| **Phase 3:** Polish | 13-16 | Testing, docs, deployment | Release + blog post |

---

## Key Milestones (Decision Points)

| Milestone | Date | Go/No-Go | If "No-Go" |
|-----------|------|----------|-----------|
| Week 1: Data collected & validated | Day 7 | Data quality good? | Collect more, adjust timeline |
| Week 4: Both models trained | Day 28 | <50px RMSE + >95% acc? | Retrain with augmentation |
| Week 6: Frame capture working | Day 42 | 30 FPS achieved? | Profile bottleneck, optimize |
| Week 12: Full pipeline integrated | Day 84 | 60 FPS target met? | Performance tuning sprint |
| Week 16: Release ready | Day 112 | Tests passing, docs complete? | Final polish, cut release |

---

## Resource Cross-Reference

### For Data Collection
- Implementation: `01_DATA_COLLECTION_GUIDE.md` (full code examples)
- Quick start: `00_GETTING_STARTED.md` (Week 1 checklist)
- Validation: Code snippets in `01_DATA_COLLECTION_GUIDE.md`

### For Machine Learning
- Data: `01_DATA_COLLECTION_GUIDE.md`
- Training: `03_LEARNING_OBJECTIVES.md` (Phase 1-4)
- Architecture: `RUST_ML_REWRITE_PLAN.md` (Section 1.3-1.5)
- Example code: `01_DATA_COLLECTION_GUIDE.md` + `03_LEARNING_OBJECTIVES.md`

### For Rust Development
- Architecture: `02_RUST_ARCHITECTURE.md` (complete design)
- Timeline: `RUST_ML_REWRITE_PLAN.md` (Section 2.1-2.7)
- Learning path: `03_LEARNING_OBJECTIVES.md` (Rust module 1-6)
- Implementation: Code in `02_RUST_ARCHITECTURE.md`

### For Learning
- Skills to develop: `03_LEARNING_OBJECTIVES.md`
- Checkpoint exercises: `03_LEARNING_OBJECTIVES.md` (per module)
- Resources: `03_LEARNING_OBJECTIVES.md` (end of document)

---

## File Sizes & Read Time

| Document | Size | Read Time | Purpose |
|----------|------|-----------|---------|
| 00_GETTING_STARTED.md | ~8 KB | 20 min | Week 1 orientation |
| RUST_ML_REWRITE_PLAN.md | ~20 KB | 1 hour | Full master plan |
| 01_DATA_COLLECTION_GUIDE.md | ~18 KB | 45 min | ML data methodology |
| 02_RUST_ARCHITECTURE.md | ~22 KB | 1 hour | System design |
| 03_LEARNING_OBJECTIVES.md | ~25 KB | 1.5 hours | Learning roadmap |
| **TOTAL** | **~93 KB** | **~4.5 hours** | Complete curriculum |

---

## How to Use These Documents

### During Project

1. **Week 1-4 (ML Phase):**
   - Keep `00_GETTING_STARTED.md` open for Week 1
   - Reference `01_DATA_COLLECTION_GUIDE.md` during data collection
   - Use `03_LEARNING_OBJECTIVES.md` (ML sections) for training
   - Check `RUST_ML_REWRITE_PLAN.md` for timeline/milestones

2. **Week 5-12 (Rust Phase):**
   - Keep `02_RUST_ARCHITECTURE.md` as reference
   - Use `03_LEARNING_OBJECTIVES.md` (Rust sections) for learning
   - Check `RUST_ML_REWRITE_PLAN.md` for phase breakdown
   - Weekly checkpoint with `03_LEARNING_OBJECTIVES.md` template

3. **Week 13-16 (Polish Phase):**
   - Use `03_LEARNING_OBJECTIVES.md` to verify all skills covered
   - Reference `02_RUST_ARCHITECTURE.md` for testing/deployment sections
   - Write blog post covering everything from `RUST_ML_REWRITE_PLAN.md`

### For Documentation

1. Copy the weekly checkpoint template from `03_LEARNING_OBJECTIVES.md`
2. Fill it out every Friday evening
3. Keep in `.planning/weekly_progress/` directory
4. Reference if you need to debug or adjust timeline

### For Interviews/Portfolio

**Elevator pitch:** (30 seconds)
> "I built IrisFlow in Rust using self-trained ML models. It's a real-time gaze tracker + gesture recognizer that controls your desktop at 60 FPS with <16ms latency, achieving <50px gaze accuracy and >95% gesture recognition."

**Deep dive:** (5 minutes)
- Data collection methodology (Python)
- Model training (PyTorch: regression + LSTM)
- Rust architecture (frame capture, inference, optimization)
- Performance results (FPS, latency breakdown, memory)

---

## When to Adjust the Plan

### Red flags that require pivot:
1. **Data quality issues** after Week 1 → Spend extra week collecting
2. **Model accuracy stagnates** (>3 weeks, <80%) → Augment data or try different architecture
3. **Performance target unreachable** (Week 12, <40 FPS) → Lower target or profile for bigger bottlenecks
4. **Rust/FFI complexity** spiraling → Simplify with ONNX Runtime instead of raw inference

### When to accelerate:
1. **Data collection done early** → Start modeling Week 2 instead of Week 3
2. **Model training done early** → Begin Rust development early
3. **Frame capture + inference complete** → Skip to optimization

### When to scope down:
1. **Running behind** (Week 10, <40 FPS) → Remove gesture recognition, focus on gaze
2. **Platform support** → Ship macOS first, Linux later
3. **Deployment** → Ship binary+Docker instead of multiple package managers

---

## Estimated Effort Breakdown

| Phase | Hours | % | Notes |
|-------|-------|---|-------|
| **Phase 1: ML** | ~120 | 30% | Data collection (40h) + training (80h) |
| **Phase 2: Rust** | ~220 | 55% | Architecture, implementation, optimization |
| **Phase 3: Polish** | ~60 | 15% | Tests, docs, blog, release |
| **TOTAL** | **~400** | **100%** | 12-16 weeks @ 25-30 hours/week |

**Your pace:** 25-30 hours/week → 13-16 weeks completion

---

## Decision Tree: Where to Start

```
START
  ↓
Have you set up Python + PyTorch? 
  → NO: Read 00_GETTING_STARTED.md (Days 1-2)
  → YES: Move to next
  ↓
Do you understand the full project scope?
  → NO: Read RUST_ML_REWRITE_PLAN.md (30 min)
  → YES: Move to next
  ↓
Ready to collect data?
  → YES: Read 01_DATA_COLLECTION_GUIDE.md + start Week 1
  → NO: Read 03_LEARNING_OBJECTIVES.md to understand ML concepts first
  ↓
Ready to understand Rust architecture?
  → YES: Read 02_RUST_ARCHITECTURE.md
  → NO: Wait until Week 5, or read in Week 4 (end of Phase 1)
  ↓
Ready to start coding?
  → DATA: Week 1 tasks in 00_GETTING_STARTED.md
  → RUST: Week 5 tasks in RUST_ML_REWRITE_PLAN.md (Section 2.1)
```

---

## Support & Troubleshooting

### Common Questions

**Q: I don't know how to use ONNX Runtime in Rust**
→ See `02_RUST_ARCHITECTURE.md` (Module 2.3: Model Loading + Inference)

**Q: My gaze model isn't converging**
→ See `03_LEARNING_OBJECTIVES.md` (Phase 2: Regression) for debugging tips

**Q: 60 FPS target seems unrealistic**
→ See `02_RUST_ARCHITECTURE.md` (Performance Targets) for realistic numbers + optimization strategies

**Q: How do I know if my data is good?**
→ See `01_DATA_COLLECTION_GUIDE.md` (Part 3: Data Validation & Cleaning)

**Q: I'm stuck on unsafe Rust**
→ See `03_LEARNING_OBJECTIVES.md` (Module 2: Unsafe Rust & FFI) + links to resources

**Q: How much time should this actually take?**
→ See this file (Estimated Effort Breakdown) + `RUST_ML_REWRITE_PLAN.md` (Timeline & Milestones)

---

## Version & Updates

**Document version:** 1.0  
**Last updated:** 2026-04-17  
**Status:** Ready for Week 1

To keep this current:
- Update status after each phase completion
- Add weekly progress checkpoints
- Adjust timeline based on actual velocity
- Document key learnings/blockers

---

## Next Step

**👉 Start here:** Open `00_GETTING_STARTED.md` and follow Week 1 checklist.

Good luck! 🚀

