# FINTECH_CUSTOMER_SEGMENTATION
Fintech User Segmentation (Unsupervised Clustering)
Segments a fintech app's user base into behavioral personas using only transaction and engagement data with no labels, no surveys. This is the kind of analysis a growth or product team runs to decide which features, offers, or nudges to build for which users. Unlike the loan-default and stock-prediction projects, this one is unsupervised which is a different core ML skill.

Note on data: synthetic (2,500 users) but built from five realistic underlying behavioral archetypes with natural overlap and noise — close to what a real neobank/wallet app's event + transaction logs would look like. The pipeline runs unchanged on real user data with the same columns (income, spend, savings rate, app engagement, transaction habits, investment/credit balances).

1. Features Used
Ten behavioral and financial signals per user: age, monthly income, savings rate, % of spend that's discretionary, app opens/week, average transaction value, transactions/month, investment balance, credit utilization, and monthly spend.

2. Key Preprocessing Decision (and why it mattered)
* Monetary features (income, investment balance, transaction value) are heavily right-skewed and a handful of high-net-worth users stretch the scale far beyond everyone else.
* Run K-means on standardized-but-untransformed data and it collapses to k=2: "high income" vs. "everyone else", because income dominates the distance metric.
* Log-transforming the skewed monetary features before scaling fixed this and revealed a much more useful k=4 structure (chosen by silhouette score, see plots/04_k_selection.png). This is a standard but easy-to-miss step, worth calling out explicitly since it changes the business usefulness of the result entirely.

3. The Four Segments Found
The data below is grouped according to Segment, Size, Age, Income, Savings, Rate and Key Trait
* Steady Mainstream, 1,181 (47%), 36, $5,521, 18%,	Balanced spend/save, moderate engagement
* Young Digital Spender, 551 (22%),	23,	$3,337,	4%,	Highest app engagement (21 opens/wk), high discretionary spend, near-zero investment
* High-Income Investor,	349 (14%), 46, $13,184,	30%, Largest transactions, $70K+ avg investment balance, lowest credit use
* Credit-Reliant Household,	419 (17%), 41, $3,718, -1%,	76% credit utilization, negative savings rate, low investment
(Full numeric profile in cluster_profiles.csv; visual comparison in plots/06_cluster_profile_heatmap.png.)

A sanity-check cross-tab against the five true generating personas (plots/07_cluster_vs_true_persona.png) confirms the clustering recovered four of the five cleanly and it only merged "Budget-Conscious Saver" and "Steady Middle," two personas that were behaviorally similar by design. That's a reasonable thing for the algorithm to do, not a failure of it.

4. Business Read on Each Segment
* Steady Mainstream : the largest group; good target for standard cross-sell (higher-yield savings, round-up investing) without a heavy engagement push.
* Young Digital Spender : high engagement but almost no investment balance; the clearest opportunity for a "start investing with $5" micro-investing product, or budgeting nudges given the low savings rate.
* High-Income Investor : low app engagement despite high value; suggests this segment uses the app transactionally rather than as a primary financial hub, a premium/wealth-management tier could deepen engagement.
* Credit-Reliant Household : negative savings rate and high credit utilization signal financial stress; the segment most likely to benefit from (and most sensitive to) responsible-lending features like spend-alerts or a debt-paydown tool, not more credit offers.
5. How to Extend
* Try hierarchical clustering or DBSCAN and compare stability against K-means
* Track segment membership over time like persona transitions (e.g. Young Digital Spender → Steady Mainstream) are often more actionable than a single snapshot
* Feed cluster labels into a supervised model predicting a business outcome (e.g. investment product adoption) to test whether segments actually differ in the metric that matters

How to Run
1. pip install pandas numpy scikit-learn matplotlib seaborn
2. python src/01_generate_data.py
3. python src/02_eda.py
4. python src/03_clustering.py
