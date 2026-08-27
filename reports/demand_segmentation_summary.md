# Demand Segmentation Summary

## Scope

This phase classifies item-store demand series without building the full long sales table. Metrics are computed from the wide sales matrix using vectorized operations.

## Output Dimensions

- demand_segmentation: `30,490` rows x `25` columns
- segment_summary: `4` rows x `9` columns
- abc_xyz_summary: `5` rows x `9` columns
- demand_pattern_by_category: `12` rows x `8` columns
- demand_pattern_by_store: `40` rows x `8` columns

## Syntetos-Boylan Demand Pattern Distribution

| demand_pattern | series_count | total_demand | avg_mean_demand | avg_zero_demand_pct | avg_adi | avg_cv_squared | series_share_pct | demand_share_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intermittent | 23102 | 28045633 | 0.63460094 | 73.6607 | 7.4712534 | 0.28692138 | 75.76910462446705 | 42.690400176974315 |
| Lumpy | 5911 | 18100365 | 1.6007054 | 60.18525 | 4.0957937 | 0.7659088 | 19.38668415874057 | 27.551948112538575 |
| Smooth | 980 | 13074239 | 6.9738946 | 13.762014 | 1.1675189 | 0.36067525 | 3.2141685798622497 | 19.901297821283066 |
| Erratic | 497 | 6475172 | 6.810515 | 16.807484 | 1.2068906 | 0.7262013 | 1.6300426369301408 | 9.856353889204039 |

## ABC Distribution

| abc_class | series_count | total_demand | series_share_pct | demand_share_pct |
| --- | --- | --- | --- | --- |
| A | 9202 | 52555695 | 30.180387012135125 | 79.99903768009116 |
| B | 9754 | 9854630 | 31.990816661200395 | 15.000485041504193 |
| C | 11534 | 3285084 | 37.82879632666448 | 5.000477278404645 |

## XYZ Distribution

| xyz_class | series_count | total_demand | series_share_pct | demand_share_pct |
| --- | --- | --- | --- | --- |
| X | 77 | 3526955 | 0.25254181698917677 | 5.368647602148272 |
| Y | 1630 | 17143530 | 5.346015086913742 | 26.09547647385832 |
| Z | 28783 | 45024924 | 94.40144309609708 | 68.5358759239934 |

## ABC-XYZ Matrix

| abc_class | xyz_class | abc_xyz_segment | series_count | total_demand | avg_zero_demand_pct | avg_coefficient_of_variation | series_share_pct | demand_share_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | X | AX | 77 | 3526955 | 0.8241628 | 0.41947472 | 0.25254181698917677 | 5.368647602148272 |
| A | Y | AY | 1630 | 17143530 | 18.41193 | 0.8528723 | 5.346015086913742 | 26.09547647385832 |
| A | Z | AZ | 7495 | 31885210 | 46.21605 | 1.4197935 | 24.581830108232207 | 48.53491360408457 |
| B | Z | BZ | 9754 | 9854630 | 69.48984 | 1.9670618 | 31.990816661200395 | 15.000485041504193 |
| C | Z | CZ | 11534 | 3285084 | 88.87085 | 3.5971684 | 37.82879632666448 | 5.000477278404645 |

## Principal Category-Pattern Combinations

| cat_id | demand_pattern | series_count | total_demand | avg_zero_demand_pct | avg_adi | avg_cv_squared | dimension_demand_share_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FOODS | Intermittent | 9195 | 15566504 | 69.42123 | 5.9852986 | 0.34676662 | 34.52323144637654 |
| FOODS | Lumpy | 4099 | 13541134 | 58.022083 | 3.5056026 | 0.7116969 | 30.031386824453232 |
| FOODS | Smooth | 710 | 10956791 | 12.725533 | 1.1542116 | 0.35415745 | 24.29985766891368 |
| HOUSEHOLD | Intermittent | 9373 | 9842997 | 74.4932 | 7.689678 | 0.25744134 | 67.97335344290009 |
| FOODS | Erratic | 366 | 5025510 | 16.526983 | 1.2032276 | 0.7062907 | 11.145524060256546 |
| HOBBIES | Lumpy | 1046 | 2788243 | 67.058136 | 5.895875 | 0.90240145 | 45.52382118599791 |
| HOBBIES | Intermittent | 4534 | 2636132 | 80.53736 | 10.033242 | 0.22649772 | 43.04029519331243 |
| HOUSEHOLD | Smooth | 241 | 1945718 | 16.41119 | 1.2014798 | 0.3759038 | 13.436657281741798 |
| HOUSEHOLD | Lumpy | 766 | 1770988 | 62.37554 | 4.7959356 | 0.86962044 | 12.230014218955338 |
| HOUSEHOLD | Erratic | 90 | 920967 | 16.309462 | 1.197692 | 0.76933414 | 6.359975056402776 |
| HOBBIES | Erratic | 41 | 528695 | 20.404676 | 1.259781 | 0.80925804 | 8.63203696447231 |
| HOBBIES | Smooth | 29 | 171730 | 17.122412 | 1.2110906 | 0.39369485 | 2.803846656217346 |

## Principal Store-Pattern Combinations

| store_id | demand_pattern | series_count | total_demand | avg_zero_demand_pct | avg_adi | avg_cv_squared | dimension_demand_share_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CA_3 | Intermittent | 2049 | 3876712 | 67.35018 | 5.639726 | 0.32244062 | 34.650068196972164 |
| CA_1 | Intermittent | 2295 | 3498511 | 69.16864 | 5.6633053 | 0.30719122 | 45.445737038295626 |
| CA_3 | Smooth | 223 | 3282801 | 14.563326 | 1.1774282 | 0.35049117 | 29.341689175540616 |
| WI_2 | Intermittent | 2361 | 2957723 | 75.698006 | 8.376412 | 0.2739701 | 45.197395726046956 |
| TX_2 | Intermittent | 2184 | 2778108 | 72.47998 | 6.475935 | 0.29704154 | 38.507903100250836 |
| WI_1 | Intermittent | 2480 | 2733380 | 72.92685 | 6.4640174 | 0.2864084 | 53.08500849280898 |
| TX_3 | Intermittent | 2346 | 2655030 | 75.04203 | 7.741881 | 0.27520218 | 43.60134858843255 |
| CA_3 | Lumpy | 685 | 2500101 | 56.967728 | 3.6857562 | 0.74529123 | 22.345913276332702 |
| WI_3 | Intermittent | 2331 | 2488182 | 76.338646 | 8.554526 | 0.26781392 | 38.70980689762036 |
| TX_1 | Intermittent | 2316 | 2419572 | 76.34912 | 8.308808 | 0.27749288 | 43.24299786320356 |
| CA_2 | Intermittent | 2189 | 2378435 | 74.53821 | 7.9750357 | 0.3142566 | 41.83353193884416 |
| WI_2 | Lumpy | 590 | 2305880 | 60.192524 | 4.064749 | 0.7269048 | 35.23648795265045 |

## Forecasting Implications

- Smooth and high-volume series are the best starting point for baseline and machine learning forecasting.
- Erratic and Lumpy series require robust error metrics and careful treatment of spikes.
- Intermittent series may need specialized intermittent-demand baselines before advanced ML.
- ABC-XYZ segments help decide where model complexity is worth the operational cost.

## Inventory Implications

- AX and AY series should receive the highest replenishment planning attention because they combine high volume with more predictable demand.
- AZ series are high-value but volatile; safety stock and exception monitoring matter more than pure point forecasts.
- C segments contain many low-volume series and may be better managed through simpler inventory rules.

## Limitations

- Zero demand can mean no customer demand, stockout, product unavailability, or pre-launch periods.
- ADI and CV² are historical descriptors, not causal explanations.
- The analysis uses `sales_train_validation.csv`; future phases should handle validation/evaluation horizons explicitly.
- XYZ thresholds are business rules based on CV: X < 0.50, Y < 1.00, Z >= 1.00. They can be tuned with planner feedback.

## Runtime And Memory

- Execution time: `18.44 seconds`
- Peak traced Python allocation: `895.98 MB`
