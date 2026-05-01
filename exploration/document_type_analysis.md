# Document Type Analysis — Knowledge Base Corpus

> Findings from subagent exploration of ~38 sampled files across 774 total markdown documents.

---

## Document Type Distribution (all companies)

| Document Type | Count (sampled) | % | Best suited for |
|---|---|---|---|
| **How-To Guide** | 14 | 37% | "How do I..." step-by-step queries |
| **Reference** | 6 | 16% | "What are the options / limits / features?" |
| **Conceptual/Overview** | 5 | 13% | Background context, product introductions |
| **Integration Guide** | 4 | 11% | Third-party ATS/SSO setup walkthroughs |
| **FAQ** | 4 | 11% | Direct Q&A mapping |
| **Release Notes** | 2 | 5% | "What's new?" — poor retrieval fit |
| **Troubleshooting** | 2 | 5% | Error/problem diagnosis |
| **Policy/Legal** | 1 | 3% | Compliance — escalate, do not answer directly |

**Directly actionable for agent responses (How-To + FAQ + Reference): ~64% of corpus**

---

## Per-Company Breakdown

### HackerRank (24 of 438 files sampled)
- Dominant type: **How-To Guide** (37.5%) — UI-centric, step-by-step navigation
- Strong integration focus (12.5%) — ATS, SSO, SCIM provisioning
- Communicates changes via dated **Release Notes**
- Audience: recruiters and admins

### Claude (11 of 322 files sampled)
- Balanced mix: How-To (45%) + FAQ (18%) + Conceptual (18%)
- Developer-friendly tone; strong API/technical content
- Includes **Policy/Legal** (9%) — data processing, compliance
- Audience: developers, enterprise IT admins, end users

### Visa (3 of 14 files sampled)
- Dominant type: **Reference** (67%) — rules, fees, regulations
- Oldest content in corpus (2017–2019 vs. 2025–2026 for others)
- Compliance and risk-focused; merchant-centric
- Thin corpus — expect more escalations for Visa tickets

---

## Structural Signals by Document Type

| Type | Key signals to detect |
|---|---|
| **How-To Guide** | Action-verb H1 ("Set Up", "Install", "Configure"), numbered steps, Prerequisites section |
| **FAQ** | Explicit "FAQ" / "FAQs" heading, Q&A pairs, "How do I...?" / "Why...?" format |
| **Reference** | Tables, lists of options, endpoint schemas, config parameter rows |
| **Conceptual/Overview** | "Introduction to", "What is", key benefits/features listed, no procedural steps |
| **Integration Guide** | Third-party product name in title, cross-system prerequisites, API credential tables |
| **Troubleshooting** | "If X then Y" logic, error messages, checklists, incident response steps |
| **Release Notes** | Dated headings, feature announcements, deprecation notices |
| **Policy/Legal** | DPA, ToS, GDPR, compliance, regulatory language |

---

## Implications for Agent Design

### Query → Document Type Routing

| User query pattern | Target doc type | Expected confidence |
|---|---|---|
| "How do I...", "How to..." | How-To Guide | High |
| "What is...", "Tell me about..." | Conceptual → How-To chain | Medium |
| "What are the options / limits?" | Reference | High |
| "Why can't I...", "Error..." | FAQ or Troubleshooting | Medium |
| "Is it possible to...", "Does X support..." | FAQ or Conceptual | Medium |
| Legal / compliance / data questions | Policy/Legal | **Escalate** |
| Third-party integration setup | Integration Guide | Medium (complex) |

### Escalation Triggers by Doc Type

- **Always escalate**: Policy/Legal content — never synthesize compliance answers
- **Escalate if thin match**: Visa corpus is too small (14 files) for confident grounding
- **Provide overview + escalate**: Integration Guides — acknowledge the integration exists, flag complexity for human follow-up

### Retrieval Strategy

1. Classify the query intent first (procedural / factual / conceptual / complaint)
2. Route to the matching document type(s) before doing vector search
3. Use Conceptual/Overview docs as context enrichment, not as the primary answer
4. Release Notes are poor retrieval targets — exclude from primary search index unless query explicitly asks about recent changes
