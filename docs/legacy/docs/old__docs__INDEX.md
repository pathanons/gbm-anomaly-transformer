# 📚 Documentation Index

> Complete guide to all documentation files organized by category

---

## 🗂️ Documentation Structure

```
docs/
├── guides/                          ← How-to guides and tutorials
├── research/                        ← Research framework and methodology  
├── specifications/                  ← Technical specifications and contracts
├── planning/                        ← Project planning and roadmaps
├── summaries/                       ← Executive summaries and reports
├── archived/                        ← Legacy and archived documentation
└── INDEX.md                         ← This file
```

---

## 📖 Guides & Tutorials

**Location**: `docs/guides/`

### [ADVANCED_TIMESERIES_GUIDE.md](guides/ADVANCED_TIMESERIES_GUIDE.md)
- Advanced time-series analysis methods
- 6 statistical analyses (stationarity, extremes, OHLCV, volume, normality, rolling)
- Expected findings and interpretations
- Integration with other EDA scripts

---

## 🔬 Research Framework

**Location**: `docs/research/`

### [RESEARCH_FRAMEWORK.md](research/RESEARCH_FRAMEWORK.md)
- 4 main research hypotheses (H1-H4)
- 4-week research timeline
- Statistical methods and validation approach
- Power-law distribution focus

### [RESEARCH_JOURNEY.md](research/RESEARCH_JOURNEY.md)
- Complete research log and progress tracking
- Weekly milestones and deliverables
- Challenges and solutions
- Current status and next steps

### [ACADEMIC_RESEARCH_DIRECTION.md](research/ACADEMIC_RESEARCH_DIRECTION.md)
- Publication-focused research direction
- Academic contributions and novelty
- Comparison with existing works
- Future research opportunities

---

## 🔧 Technical Specifications

**Location**: `docs/specifications/`

### [DESIGN_BY_CONTRACT.md](specifications/DESIGN_BY_CONTRACT.md)
- Contract principles (PRE/POST/FAIL conditions)
- Method-by-method specifications for all EDA scripts
- Data invariants and assertions
- Error handling strategy

---

## 📋 Planning & Roadmap

**Location**: `docs/planning/`

### [REORGANIZATION_PLAN.md](planning/REORGANIZATION_PLAN.md)
- Complete project restructuring plan
- Directory organization strategy
- Dependencies and build structure
- Migration timeline

### [REORGANIZATION_CHECKLIST.md](planning/REORGANIZATION_CHECKLIST.md)
- Step-by-step reorganization tasks
- Status tracking for each phase
- Completion criteria and verification

### [IMPLEMENTATION_CHECKLIST.md](planning/IMPLEMENTATION_CHECKLIST.md)
- Week-by-week implementation tasks
- Script development milestones
- Testing and validation checklist
- Deployment readiness criteria

### [IMPLEMENTATION_SUMMARY.md](planning/IMPLEMENTATION_SUMMARY.md)
- Current implementation status
- Completed components summary
- Remaining work items
- Resource requirements

---

## 📊 Executive Summaries & Reports

**Location**: `docs/summaries/`

### [EXECUTIVE_SUMMARY.md](summaries/EXECUTIVE_SUMMARY.md)
- High-level project overview
- Key objectives and hypotheses
- Current status snapshot
- Progress metrics

### [VISUAL_OVERVIEW.md](summaries/VISUAL_OVERVIEW.md)
- ASCII diagrams and flowcharts
- Data pipeline visualization
- Architecture overview
- Process flow diagrams

### [DOCUMENTATION_HUB.md](summaries/DOCUMENTATION_HUB.md)
- Central reference for all documentation
- Quick links to key sections
- Document cross-references
- FAQ and troubleshooting

### [WEEK1_EDA_SCRIPTS_READY.md](summaries/WEEK1_EDA_SCRIPTS_READY.md)
- Week 1 EDA completion status
- Scripts ready for execution
- Output file locations
- Next steps for Week 2

### [CLEANUP_AND_RESTRUCTURE.md](summaries/CLEANUP_AND_RESTRUCTURE.md)
- Summary of project cleanup efforts
- Reorganization completed
- File structure improvements
- Code quality enhancements

---

## 📦 Archived Documentation

**Location**: `docs/archived/`

> Legacy documentation maintained for reference

### [BATCH_NORMALIZATION_GUIDE.md](archived/BATCH_NORMALIZATION_GUIDE.md)
- Batch normalization implementation guide
- Training techniques and parameters
- Performance tuning recommendations

### [POWERLAW_PRIOR_GUIDE.md](archived/POWERLAW_PRIOR_GUIDE.md)
- Power-law prior implementation
- Bayesian modeling approach
- Parameter tuning guide

### [TRAINING_GUIDE.md](archived/TRAINING_GUIDE.md)
- Model training procedures
- Hyperparameter tuning
- Training loop implementation
- Validation strategies

### [PROJECT_STATUS_REPORT.md](archived/PROJECT_STATUS_REPORT.md)
- Previous status snapshots
- Historical progress tracking
- Archived milestones

### [QUICK_REFERENCE.md](archived/QUICK_REFERENCE.md)
- Quick lookup reference
- Command cheatsheet
- Common tasks and solutions

---

## 🚀 Quick Start

### 1️⃣ Understand the Project
Start here: [RESEARCH_FRAMEWORK.md](research/RESEARCH_FRAMEWORK.md)  
Then read: [EXECUTIVE_SUMMARY.md](summaries/EXECUTIVE_SUMMARY.md)

### 2️⃣ Learn About Implementation
Review: [DESIGN_BY_CONTRACT.md](specifications/DESIGN_BY_CONTRACT.md)  
Check status: [IMPLEMENTATION_SUMMARY.md](planning/IMPLEMENTATION_SUMMARY.md)

### 3️⃣ Use the Scripts
Guide: [ADVANCED_TIMESERIES_GUIDE.md](guides/ADVANCED_TIMESERIES_GUIDE.md)  
Status: [WEEK1_EDA_SCRIPTS_READY.md](summaries/WEEK1_EDA_SCRIPTS_READY.md)

### 4️⃣ Track Progress
Timeline: [RESEARCH_JOURNEY.md](research/RESEARCH_JOURNEY.md)  
Checklist: [IMPLEMENTATION_CHECKLIST.md](planning/IMPLEMENTATION_CHECKLIST.md)

---

## 📂 Repository Structure

```
Anomaly-Transformer/
├── README.md                        ← Start here
├── docs/                            ← All documentation (organized)
│   ├── guides/
│   ├── research/
│   ├── specifications/
│   ├── planning/
│   ├── summaries/
│   ├── archived/
│   └── INDEX.md                     ← This file
│
├── scripts/                         ← Analysis scripts
│   └── research/
│       ├── 01_exploratory_data_analysis.py
│       ├── 02_temporal_pattern_analysis.py
│       ├── 03_advanced_timeseries_analysis.py
│       ├── run_single_ticker_eda.py
│       ├── run_all_tickers_eda.py
│       ├── run_industry_group_eda.py
│       └── run_all_eda.py
│
├── src/                             ← Source code
│   ├── model/                       ← Model implementations
│   ├── data_factory/                ← Data loading and preprocessing
│   └── utils/                       ← Utility functions
│
├── datasets/                        ← Data files
├── results/                         ← Experiment results
├── checkpoints/                     ← Model checkpoints
├── requirements.txt                 ← Python dependencies
└── LICENSE
```

---

## 🔍 Document Purpose Matrix

| Document | Purpose | Audience | Update Freq |
|----------|---------|----------|------------|
| RESEARCH_FRAMEWORK.md | Define hypothesis & methodology | Researchers | Monthly |
| DESIGN_BY_CONTRACT.md | Technical specifications | Developers | As needed |
| IMPLEMENTATION_CHECKLIST.md | Track completion | Project manager | Weekly |
| ADVANCED_TIMESERIES_GUIDE.md | Usage guide | All users | As needed |
| RESEARCH_JOURNEY.md | Progress log | Team | Daily |
| EXECUTIVE_SUMMARY.md | Status snapshot | Stakeholders | Weekly |
| WEEK1_EDA_SCRIPTS_READY.md | Script readiness | Users | One-time |

---

## 💡 How to Navigate

### By Role

**👨‍💼 Project Manager**
1. [EXECUTIVE_SUMMARY.md](summaries/EXECUTIVE_SUMMARY.md)
2. [IMPLEMENTATION_CHECKLIST.md](planning/IMPLEMENTATION_CHECKLIST.md)
3. [RESEARCH_JOURNEY.md](research/RESEARCH_JOURNEY.md)

**👨‍🔬 Researcher**
1. [RESEARCH_FRAMEWORK.md](research/RESEARCH_FRAMEWORK.md)
2. [ACADEMIC_RESEARCH_DIRECTION.md](research/ACADEMIC_RESEARCH_DIRECTION.md)
3. [RESEARCH_JOURNEY.md](research/RESEARCH_JOURNEY.md)

**👨‍💻 Developer**
1. [DESIGN_BY_CONTRACT.md](specifications/DESIGN_BY_CONTRACT.md)
2. [IMPLEMENTATION_SUMMARY.md](planning/IMPLEMENTATION_SUMMARY.md)
3. [ADVANCED_TIMESERIES_GUIDE.md](guides/ADVANCED_TIMESERIES_GUIDE.md)

**🎓 New Team Member**
1. [README.md](../README.md)
2. [VISUAL_OVERVIEW.md](summaries/VISUAL_OVERVIEW.md)
3. [RESEARCH_FRAMEWORK.md](research/RESEARCH_FRAMEWORK.md)
4. [WEEK1_EDA_SCRIPTS_READY.md](summaries/WEEK1_EDA_SCRIPTS_READY.md)

### By Task

**Get Started Quickly**
→ [WEEK1_EDA_SCRIPTS_READY.md](summaries/WEEK1_EDA_SCRIPTS_READY.md)

**Understand the Theory**
→ [RESEARCH_FRAMEWORK.md](research/RESEARCH_FRAMEWORK.md)

**Learn How to Use Scripts**
→ [ADVANCED_TIMESERIES_GUIDE.md](guides/ADVANCED_TIMESERIES_GUIDE.md)

**Track Project Progress**
→ [RESEARCH_JOURNEY.md](research/RESEARCH_JOURNEY.md)

**Check Implementation Status**
→ [IMPLEMENTATION_SUMMARY.md](planning/IMPLEMENTATION_SUMMARY.md)

---

## 📞 Documentation Maintenance

- **Last Updated**: March 2026
- **Maintainer**: Research Team
- **Review Cycle**: Monthly
- **Archive Policy**: Moved to `archived/` after 3 months of inactivity

---

## 🔗 External References

- **Paper**: 2110.02642v5.pdf (Anomaly Transformer original)
- **Code Repository**: scripts/research/
- **Data**: datasets/SP500/
- **Results**: results/

---

*For questions or suggestions about documentation, see [DOCUMENTATION_HUB.md](summaries/DOCUMENTATION_HUB.md)*
