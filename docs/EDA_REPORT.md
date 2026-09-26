\# Amazon ML Challenge 2026 — Dataset EDA Report



\## 1. Overview



This report documents the exploratory data analysis (EDA) performed on the Amazon ML Challenge 2026 training datasets.



The analysis covers:



\- Source 1

\- Source 2

\- Source 3

\- Ground-truth matching data

\- Column structure and data types

\- Missing values

\- Duplicate entity IDs

\- Repeated business names

\- Repeated addresses

\- Country distribution

\- Ground-truth matching patterns



The datasets are large TSV files, so the analysis was performed using pandas chunk processing with a chunk size of 100,000 rows.



\---



\## 2. Dataset Structure



All three source datasets contain the following columns:



| Column | Description |

|---|---|

| `entity\_id` | Unique entity identifier for the source |

| `business\_name` | Business or organization name |

| `business\_address` | Business address |

| `country` | Country associated with the entity |



The ground-truth dataset contains:



| Column | Description |

|---|---|

| `source1\_entity\_id` | Entity identifier from Source 1 |

| `matched\_entity\_ids` | Comma-separated matching entity IDs from Source 2 and Source 3 |



All observed source columns were read as string (`object`-like string data through pandas) to preserve identifiers and text values exactly.



\---



\# 3. Source 1 Analysis



\*\*File:\*\* `data/train/train\_source1.tsv`



\### Basic statistics



\- Total rows: \*\*2,206,821\*\*

\- Unique entity IDs: \*\*2,206,821\*\*

\- Unique business names: \*\*1,538,804\*\*

\- Unique addresses: \*\*2,130,606\*\*

\- Duplicate entity IDs: \*\*0\*\*

\- Duplicate rows detected within chunks: \*\*0\*\*



\### Missing values



| Column | Missing/Empty |

|---|---:|

| `entity\_id` | 0 |

| `business\_name` | 0 |

| `business\_address` | 0 |

| `country` | 0 |



Source 1 therefore has no missing or empty values in the four expected columns.



\### Business-name patterns



There are \*\*668,017 repeated business-name occurrences beyond the first occurrence\*\*.



The most frequently repeated normalized names include:



\- `primary care group`

\- `ear nose \& throat group`

\- `pediatric group`

\- `womens health group`

\- `physical therapy group`



This indicates that business names are not always unique identifiers and that common healthcare/business category terms occur repeatedly.



\### Address patterns



There are \*\*76,215 repeated address occurrences beyond the first occurrence\*\*.



Repeated addresses occur in multiple records, indicating that address information can be shared by multiple entities.



\### Country distribution



| Country | Rows |

|---|---:|

| US | 1,323,633 |

| India | 883,188 |



Source 1 therefore contains records from both the US and India, with the US representing the larger portion of the dataset.



\---



\# 4. Source 2 Analysis



\*\*File:\*\* `data/train/train\_source2.tsv`



\### Basic statistics



\- Total rows: \*\*5,034,616\*\*

\- Unique entity IDs: \*\*5,034,616\*\*

\- Unique business names: \*\*4,192,965\*\*

\- Unique addresses: \*\*4,300,098\*\*

\- Duplicate entity IDs: \*\*0\*\*

\- Duplicate rows detected within chunks: \*\*0\*\*



\### Missing values



| Column | Missing/Empty |

|---|---:|

| `entity\_id` | 0 |

| `business\_name` | 0 |

| `business\_address` | 168,967 |

| `country` | 0 |



The main missing-data issue in Source 2 is the business address field.



\### Business-name patterns



There are \*\*841,651 repeated business-name occurrences beyond the first occurrence\*\*.



Common repeated normalized names include:



\- `primary care`

\- `physical therapy`

\- `womens health`

\- `urgent care`

\- `behavioral health`

\- `pediatric dentistry`

\- `internal medicine`

\- `pediatric dental`

\- `earnosethroat.com`



This shows that common service or specialty terms occur frequently and should not be treated as unique identifiers by themselves.



\### Address patterns



There are \*\*565,551 repeated address occurrences beyond the first occurrence\*\*.



Some addresses occur many times across the source, showing that address information may be shared across multiple entities.



\### Country distribution



| Country | Rows |

|---|---:|

| US | 3,016,817 |

| India | 2,017,799 |



Source 2 has a larger number of records than Source 1 while maintaining the same two-country distribution.



\---



\# 5. Source 3 Analysis



\*\*File:\*\* `data/train/train\_source3.tsv`



\### Basic statistics



\- Total rows: \*\*5,285,603\*\*

\- Unique entity IDs: \*\*5,285,603\*\*

\- Unique business names: \*\*4,473,473\*\*

\- Unique addresses: \*\*4,631,187\*\*

\- Duplicate entity IDs: \*\*0\*\*

\- Duplicate rows detected within chunks: \*\*0\*\*



\### Missing values



| Column | Missing/Empty |

|---|---:|

| `entity\_id` | 0 |

| `business\_name` | 0 |

| `business\_address` | 175,916 |

| `country` | 0 |



As with Source 2, the main missing-data issue is in the business address field.



\### Business-name patterns



There are \*\*812,130 repeated business-name occurrences beyond the first occurrence\*\*.



Frequently repeated normalized names include:



\- `primary care`

\- `physical therapy`

\- `pediatric dental`

\- `urgent care`

\- `womens health`

\- `pediatric dentistry`

\- `internal medicine`

\- `behavioral health`

\- `pediatric`

\- `earnosethroat.com`



These repeated names indicate that business-name matching needs to consider additional attributes instead of relying only on exact name equality.



\### Address patterns



There are \*\*478,500 repeated address occurrences beyond the first occurrence\*\*.



Some repeated addresses are incomplete or location-level strings, such as:



\- `ground floor, bangalore, ka`

\- `floor, mumbai, mh`

\- `18, kolkata, howrah, wb`



This indicates that address quality and completeness vary across records.



\### Country distribution



| Country | Rows |

|---|---:|

| US | 3,170,056 |

| India | 2,115,547 |



Source 3 has the largest number of records among the three source datasets.



\---



\# 6. Cross-Source Observations



The three source datasets show several common patterns.



\### Entity IDs



All three sources have:



\- No duplicate entity IDs

\- Number of unique entity IDs equal to the number of rows



This indicates that `entity\_id` is unique within each source.



\### Business names



Repeated business names are common in all three sources.



This is important because identical or similar business names can correspond to different entities.



Therefore, business name alone should not be treated as a definitive matching key.



\### Addresses



Addresses are also repeated across records.



Source 2 and Source 3 additionally contain a substantial number of missing/empty addresses.



This means address information can be useful for matching but should be combined with other attributes.



\### Country



All three datasets contain records from:



\- United States (`US`)

\- India



The approximate distribution is consistently around 60% US and 40% India across the three sources.



\---



\# 7. Ground-Truth Analysis



\*\*File:\*\* `data/train/train\_ground\_truth.tsv`



\### Structure



The ground-truth dataset contains:



\- `2,206,821` Source 1 records

\- `source1\_entity\_id`

\- `matched\_entity\_ids`



The `matched\_entity\_ids` field contains comma-separated Source 2 and Source 3 entity IDs.



Example structure:



```text

S1-965667    S2-681193310,S2-743505751,S3-775321672,S3-11291185

