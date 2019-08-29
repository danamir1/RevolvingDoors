# RevolvingDoors
Needle in The Haystack Final Project code repo.

Files included:
- analyze_interests.py - code for analysis and visualization of interests map graph
- association_rules.py - extraction of association rules from Ynet and committees data with apriori
- committees_guests.py - extraction and cleaning of committees guests data
- extract_ynet_data.py - extraction and cleaning of Ynet articles data
- match_entities.py - code for matching entities form Ynet and committees to entities database
- maya_helper.py - cleaning and analysis of appointments data from Maya
- name_merging_tools.py - code for finding merge sets of names of entities from data
- utils.py - general io utils
- ynet_scraping.py - code for scraping articles from ynet economy channel

- visualization - interactive graphs used to analyze the interests map. each html includes the whole graph
plus separate graph for each cluster. Beware - all graphs are manipulated together with the toolbar on the top.

Each visualization name is structured as <data used>_<similarity metric>_<clustering method>_<number of
clusters if it's not 20>.html


