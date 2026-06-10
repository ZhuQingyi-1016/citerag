# Retrieval Evaluation Report

## Overall

- Passed: **47/48**

## By Retrieval Mode + Rerank

| Mode | Passed | Avg Citations | Avg Hits |
|---|---:|---:|---:|
| bm25 + none | 8/8 | 2.00 | 2.00 |
| bm25 + cohere | 8/8 | 2.00 | 2.00 |
| vector + none | 8/8 | 2.00 | 2.00 |
| vector + cohere | 8/8 | 2.00 | 2.00 |
| hybrid + none | 8/8 | 2.00 | 2.00 |
| hybrid + cohere | 7/8 | 2.00 | 2.00 |

## By Category

| Category | Passed |
|---|---:|
| keyword | 12/12 |
| semantic | 11/12 |
| confusion | 12/12 |
| no_answer | 12/12 |

## Failure Types

| Failure Type | Count |
|---|---:|
| quality_failure | 1 |

## Detailed Results

| Case | Category | Mode | Rerank | top_k | retrieve_top_k | Status | Passed | Citations | Hits |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| keyword_readme_retrieval_modes | keyword | bm25 | none | 3 | 10 | 200 | True | 3 | 3 |
| keyword_readme_retrieval_modes | keyword | bm25 | cohere | 3 | 10 | 200 | True | 3 | 3 |
| keyword_readme_retrieval_modes | keyword | vector | none | 3 | 10 | 200 | True | 3 | 3 |
| keyword_readme_retrieval_modes | keyword | vector | cohere | 3 | 10 | 200 | True | 3 | 3 |
| keyword_readme_retrieval_modes | keyword | hybrid | none | 3 | 10 | 200 | True | 3 | 3 |
| keyword_readme_retrieval_modes | keyword | hybrid | cohere | 3 | 10 | 200 | True | 3 | 3 |
| keyword_requirements_fastapi | keyword | bm25 | none | 3 | 10 | 200 | True | 1 | 1 |
| keyword_requirements_fastapi | keyword | bm25 | cohere | 3 | 10 | 200 | True | 1 | 1 |
| keyword_requirements_fastapi | keyword | vector | none | 3 | 10 | 200 | True | 1 | 1 |
| keyword_requirements_fastapi | keyword | vector | cohere | 3 | 10 | 200 | True | 1 | 1 |
| keyword_requirements_fastapi | keyword | hybrid | none | 3 | 10 | 200 | True | 1 | 1 |
| keyword_requirements_fastapi | keyword | hybrid | cohere | 3 | 10 | 200 | True | 1 | 1 |
| semantic_readme_project_capabilities | semantic | bm25 | none | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_capabilities | semantic | bm25 | cohere | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_capabilities | semantic | vector | none | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_capabilities | semantic | vector | cohere | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_capabilities | semantic | hybrid | none | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_capabilities | semantic | hybrid | cohere | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_overview | semantic | bm25 | none | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_overview | semantic | bm25 | cohere | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_overview | semantic | vector | none | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_overview | semantic | vector | cohere | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_overview | semantic | hybrid | none | 3 | 10 | 200 | True | 3 | 3 |
| semantic_readme_project_overview | semantic | hybrid | cohere | 3 | 10 | 200 | False | 3 | 3 |
| confusion_html_support_email | confusion | bm25 | none | 3 | 10 | 200 | True | 1 | 1 |
| confusion_html_support_email | confusion | bm25 | cohere | 3 | 10 | 200 | True | 1 | 1 |
| confusion_html_support_email | confusion | vector | none | 3 | 10 | 200 | True | 1 | 1 |
| confusion_html_support_email | confusion | vector | cohere | 3 | 10 | 200 | True | 1 | 1 |
| confusion_html_support_email | confusion | hybrid | none | 3 | 10 | 200 | True | 1 | 1 |
| confusion_html_support_email | confusion | hybrid | cohere | 3 | 10 | 200 | True | 1 | 1 |
| confusion_requirements_not_email | confusion | bm25 | none | 3 | 10 | 200 | True | 1 | 1 |
| confusion_requirements_not_email | confusion | bm25 | cohere | 3 | 10 | 200 | True | 1 | 1 |
| confusion_requirements_not_email | confusion | vector | none | 3 | 10 | 200 | True | 1 | 1 |
| confusion_requirements_not_email | confusion | vector | cohere | 3 | 10 | 200 | True | 1 | 1 |
| confusion_requirements_not_email | confusion | hybrid | none | 3 | 10 | 200 | True | 1 | 1 |
| confusion_requirements_not_email | confusion | hybrid | cohere | 3 | 10 | 200 | True | 1 | 1 |
| no_answer_readme_breakfast | no_answer | bm25 | none | 3 | 10 | 200 | True | 3 | 3 |
| no_answer_readme_breakfast | no_answer | bm25 | cohere | 3 | 10 | 200 | True | 3 | 3 |
| no_answer_readme_breakfast | no_answer | vector | none | 3 | 10 | 200 | True | 3 | 3 |
| no_answer_readme_breakfast | no_answer | vector | cohere | 3 | 10 | 200 | True | 3 | 3 |
| no_answer_readme_breakfast | no_answer | hybrid | none | 3 | 10 | 200 | True | 3 | 3 |
| no_answer_readme_breakfast | no_answer | hybrid | cohere | 3 | 10 | 200 | True | 3 | 3 |
| no_answer_html_ceo | no_answer | bm25 | none | 3 | 10 | 200 | True | 1 | 1 |
| no_answer_html_ceo | no_answer | bm25 | cohere | 3 | 10 | 200 | True | 1 | 1 |
| no_answer_html_ceo | no_answer | vector | none | 3 | 10 | 200 | True | 1 | 1 |
| no_answer_html_ceo | no_answer | vector | cohere | 3 | 10 | 200 | True | 1 | 1 |
| no_answer_html_ceo | no_answer | hybrid | none | 3 | 10 | 200 | True | 1 | 1 |
| no_answer_html_ceo | no_answer | hybrid | cohere | 3 | 10 | 200 | True | 1 | 1 |

## Failures

### semantic_readme_project_overview (semantic / hybrid / cohere)

- Question: `这个项目整体是做什么的？`
- Expected any of: `['Retrieval-Augmented Generation', '检索增强生成', 'RAG']`
- Answer: `这个项目整体是一个文件上传和检索系统，支持文本和Markdown文件的上传、索引和问答功能。用户可以上传文件，系统会自动索引文件内容，并允许通过关键词或语义检索来提问。`
- Failure type: `quality_failure`

## Key Takeaways

- This report compares `bm25`, `vector`, and `hybrid` retrieval modes together with `none` and `cohere` rerank modes on the local eval set.
- The baseline `none` rerank mode remained the most stable default, with all `none` configurations passing their current cases.
- After adding retry/backoff in eval and graceful fallback in reranking, infra-driven failures were eliminated from this report; the remaining failure is a summary-style semantic quality issue rather than an external API reliability issue.
- `cohere` reranking worked well in most cases, but this eval suggests reranking should remain optional and task-dependent rather than being enabled blindly by default.
- Refusal-style cases are graded with a normalized refusal matcher so semantically correct no-answer responses are not penalized for wording differences.
