# -*- coding: utf-8 -*-
"""
Creates a Parquet file containing only the static/current race information
by dropping the specified wide-format workout and past performance columns
from the full parsed data.

Reads:
- parsed_race_data_full.parquet (output of the modified bris_spec.py)

Outputs:
- current_race_info.parquet (static/current race data only)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Final, Optional
import logging  # Import logging

from config.settings import PARSED_RACE_DATA, CURRENT_RACE_INFO, PROCESSED_DATA_DIR

# --- Centralized Path Configuration ---
WIDE_DATA_FILE_PATH: Final[Path] = PARSED_RACE_DATA
CURRENT_INFO_FILE_PATH: Final[Path] = CURRENT_RACE_INFO

# COLUMNS_TO_DROP and record_groups are defined as before...
COLUMNS_TO_DROP: Final[List[str]] = [
    # Workout Columns
    'time_of_workout_1', '2_negative_time_if_a', '3_bullet_work', '4_ie_34_80_means',
    '5_a_bullet_work_in', '6_a_time_of_34_4_5', '7_pos_120', '8_pos_121', '9_pos_122',
    '10_pos_123', '11_pos_124', '12_pos_125', 'track_of_workout_1', '2_pos_127',
    '3_pos_128', '4_pos_129', '5_pos_130', '6_pos_131', '7_pos_132', '8_pos_133',
    '9_pos_134', '10_pos_135', '11_pos_136', '12_pos_137', 'distance_of_workout_1',
    '2_value_for', '3_about_distances', '4_pos_141', '5_pos_142', '6_pos_143',
    '7_pos_144', '8_pos_145', '9_pos_146', '10_pos_147', '11_pos_148', '12_pos_149',
    'track_condition_of_workout_1', '2_pos_151', '3_pos_152', '4_pos_153', '5_pos_154',
    '6_pos_155', '7_pos_156', '8_pos_157', '9_pos_158', '10_pos_159', '11_pos_160',
    '12_pos_161', 'description_of_workout_1', '2_pos_163', '3_1st_character_h_or_b',
    '4_h_for_handily_b_for_breezing', '5_pos_166', '6_2nd_character_g',
    '7_if_worked_from_gate', '8_pos_169', '9_3rd_character_d', '10_if_dogs_are_up',
    '11_pos_172', '12_pos_173', 'main_inner_track_indicator_1', '2_im_inner_dirt',
    '3_tt_training_trk', '4_t_main_turf', '5_it_inner_turf', '6_wc_wood_chip',
    '7_hc_hillside_course', '8_tn_trf_trn_trk', '9_in_inner_trf_trn',
    '10_tr_training_race', '11_if_blank_track', '12_type_unknown', 'day_distance_1',
    '2_pos_187', '3_pos_188', '4_pos_189', '5_pos_190', '6_pos_191', '7_pos_192',
    '8_pos_193', '9_pos_194', '10_pos_195', '11_pos_196', '12_pos_197',
    'other_works_that_day_dist_1', '2_pos_199', '3_pos_200', '4_pos_201', '5_pos_202',
    '6_pos_203', '7_pos_204', '8_pos_205', '9_pos_206', '10_pos_207', '11_pos_208',
    '12_pos_209',
    # Past Performance Columns
    'race_1_pos_256', 'race_2_pos_257', 'race_3_pos_258', 'race_4_pos_259', 'race_5_pos_260',
    'race_6_pos_261', 'race_7_pos_262', 'race_8_pos_263', 'race_9_pos_264', 'race_10_pos_265',
    'of_days_since_previous_race_1', 'of_days_since_previous_race_2', 'of_days_since_previous_race_3',
    'of_days_since_previous_race_4', 'of_days_since_previous_race_5', 'of_days_since_previous_race_6',
    'of_days_since_previous_race_7', 'of_days_since_previous_race_8', 'of_days_since_previous_race_9',
    # 'reserved_days_since_prev_race_for_10th_race_back_might_not_be_available',
    'track_code_1', 'track_code_2', 'track_code_3', 'track_code_4', 'track_code_5', 'track_code_6',
    'track_code_7', 'track_code_8', 'track_code_9', 'track_code_10', 'bris_track_code_1',
    'bris_track_code_2', 'bris_track_code_3', 'bris_track_code_4', 'bris_track_code_5',
    'bris_track_code_6', 'bris_track_code_7', 'bris_track_code_8', 'bris_track_code_9',
    'bris_track_code_10', 'race_1_pos_296', 'race_2_pos_297', 'race_3_pos_298', 'race_4_pos_299',
    'race_5_pos_300', 'race_6_pos_301', 'race_7_pos_302', 'race_8_pos_303', 'race_9_pos_304',
    'race_10_pos_305', 'track_condition_1', 'track_condition_2', 'track_condition_3',
    'track_condition_4', 'track_condition_5', 'track_condition_6', 'track_condition_7',
    'track_condition_8', 'track_condition_9', 'track_condition_10', 'distance_in_yards_1',
    'distance_in_yards_2', 'distance_in_yards_3', 'distance_in_yards_4', 'distance_in_yards_5',
    'distance_in_yards_6', 'distance_in_yards_7', 'distance_in_yards_8', 'distance_in_yards_9',
    'distance_in_yards_10', 'surface_1', 'surface_2', 'surface_3', 'surface_4', 'surface_5',
    'surface_6', 'surface_7', 'surface_8', 'surface_9', 'surface_10', 'special_chute_indicator_1',
    'special_chute_indicator_2', 'special_chute_indicator_3', 'special_chute_indicator_4',
    'special_chute_indicator_5', 'special_chute_indicator_6', 'special_chute_indicator_7',
    'special_chute_indicator_8', 'special_chute_indicator_9', 'special_chute_indicator_10',
    'of_entrants_1', 'of_entrants_2', 'of_entrants_3', 'of_entrants_4', 'of_entrants_5',
    'of_entrants_6', 'of_entrants_7', 'of_entrants_8', 'of_entrants_9', 'of_entrants_10',
    'post_position_1', 'post_position_2', 'post_position_3', 'post_position_4', 'post_position_5',
    'post_position_6', 'post_position_7', 'post_position_8', 'post_position_9', 'post_position_10',
    'equipment_1', 'equipment_2', 'equipment_3', 'equipment_4', 'equipment_5', 'equipment_6',
    'equipment_7', 'equipment_8', 'equipment_9', 'equipment_10', 'racename_of_previous_races_1',
    'racename_of_previous_races_2', 'racename_of_previous_races_3', 'racename_of_previous_races_4',
    'racename_of_previous_races_5', 'racename_of_previous_races_6', 'racename_of_previous_races_7',
    'racename_of_previous_races_8', 'racename_of_previous_races_9', 'racename_of_previous_races_10',
    'medication_1', 'medication_2', 'medication_3', 'medication_4', 'medication_5', 'medication_6',
    'medication_7', 'medication_8', 'medication_9', 'medication_10', 'trip_comment_1',
    'trip_comment_2', 'trip_comment_3', 'trip_comment_4', 'trip_comment_5', 'trip_comment_6',
    'trip_comment_7', 'trip_comment_8', 'trip_comment_9', 'trip_comment_10', 'winner_s_name_1',
    'winner_s_name_2', 'winner_s_name_3', 'winner_s_name_4', 'winner_s_name_5', 'winner_s_name_6',
    'winner_s_name_7', 'winner_s_name_8', 'winner_s_name_9', 'winner_s_name_10',
    '2nd_place_finishers_name_1', '2nd_place_finishers_name_2', '2nd_place_finishers_name_3',
    '2nd_place_finishers_name_4', '2nd_place_finishers_name_5', '2nd_place_finishers_name_6',
    '2nd_place_finishers_name_7', '2nd_place_finishers_name_8', '2nd_place_finishers_name_9',
    '2nd_place_finishers_name_10', '3rd_place_finishers_name_1', '3rd_place_finishers_name_2',
    '3rd_place_finishers_name_3', '3rd_place_finishers_name_4', '3rd_place_finishers_name_5',
    '3rd_place_finishers_name_6', '3rd_place_finishers_name_7', '3rd_place_finishers_name_8',
    '3rd_place_finishers_name_9', '3rd_place_finishers_name_10', 'winner_s_weight_carried_1',
    'winner_s_weight_carried_2', 'winner_s_weight_carried_3', 'winner_s_weight_carried_4',
    'winner_s_weight_carried_5', 'winner_s_weight_carried_6', 'winner_s_weight_carried_7',
    'winner_s_weight_carried_8', 'winner_s_weight_carried_9', 'winner_s_weight_carried_10',
    '2nd_place_weight_carried_1', '2nd_place_weight_carried_2', '2nd_place_weight_carried_3',
    '2nd_place_weight_carried_4', '2nd_place_weight_carried_5', '2nd_place_weight_carried_6',
    '2nd_place_weight_carried_7', '2nd_place_weight_carried_8', '2nd_place_weight_carried_9',
    '2nd_place_weight_carried_10', '3rd_place_weight_carried_1', '3rd_place_weight_carried_2',
    '3rd_place_weight_carried_3', '3rd_place_weight_carried_4', '3rd_place_weight_carried_5',
    '3rd_place_weight_carried_6', '3rd_place_weight_carried_7', '3rd_place_weight_carried_8',
    '3rd_place_weight_carried_9', '3rd_place_weight_carried_10', 'winner_s_margin_1',
    'winner_s_margin_2', 'winner_s_margin_3', 'winner_s_margin_4', 'winner_s_margin_5',
    'winner_s_margin_6', 'winner_s_margin_7', 'winner_s_margin_8', 'winner_s_margin_9',
    'winner_s_margin_10', '2nd_place_margin_1', '2nd_place_margin_2', '2nd_place_margin_3',
    '2nd_place_margin_4', '2nd_place_margin_5', '2nd_place_margin_6', '2nd_place_margin_7',
    '2nd_place_margin_8', '2nd_place_margin_9', '2nd_place_margin_10', '3rd_place_margin_1',
    '3rd_place_margin_2', '3rd_place_margin_3', '3rd_place_margin_4', '3rd_place_margin_5',
    '3rd_place_margin_6', '3rd_place_margin_7', '3rd_place_margin_8', '3rd_place_margin_9',
    '3rd_place_margin_10', 'alternate_extra_comment_line_1', 'alternate_extra_comment_line_2',
    'alternate_extra_comment_line_3', 'alternate_extra_comment_line_4', 'alternate_extra_comment_line_5',
    'alternate_extra_comment_line_6', 'alternate_extra_comment_line_7', 'alternate_extra_comment_line_8',
    'alternate_extra_comment_line_9', 'alternate_extra_comment_line_10', 'weight_1', 'weight_2',
    'weight_3', 'weight_4', 'weight_5', 'weight_6', 'weight_7', 'weight_8', 'weight_9', 'weight_10',
    'odds_1', 'odds_2', 'odds_3', 'odds_4', 'odds_5', 'odds_6', 'odds_7', 'odds_8', 'odds_9', 'odds_10',
    'entry_1', 'entry_2', 'entry_3', 'entry_4', 'entry_5', 'entry_6', 'entry_7', 'entry_8', 'entry_9',
    'entry_10', 'race_classification_1', 'race_classification_2', 'race_classification_3',
    'race_classification_4', 'race_classification_5', 'race_classification_6', 'race_classification_7',
    'race_classification_8', 'race_classification_9', 'race_classification_10',
    'claiming_price_of_horse_1', 'claiming_price_of_horse_2', 'claiming_price_of_horse_3',
    'claiming_price_of_horse_4', 'claiming_price_of_horse_5', 'claiming_price_of_horse_6',
    'claiming_price_of_horse_7', 'claiming_price_of_horse_8', 'claiming_price_of_horse_9',
    'claiming_price_of_horse_10', 'purse_1', 'purse_2', 'purse_3', 'purse_4', 'purse_5', 'purse_6',
    'purse_7', 'purse_8', 'purse_9', 'purse_10', 'start_call_position_1', 'start_call_position_2',
    'start_call_position_3', 'start_call_position_4', 'start_call_position_5', 'start_call_position_6',
    'start_call_position_7', 'start_call_position_8', 'start_call_position_9', 'start_call_position_10',
    '1st_call_position_if_any_1', '1st_call_position_if_any_2', '1st_call_position_if_any_3',
    '1st_call_position_if_any_4', '1st_call_position_if_any_5', '1st_call_position_if_any_6',
    '1st_call_position_if_any_7', '1st_call_position_if_any_8', '1st_call_position_if_any_9',
    '1st_call_position_if_any_10', '2nd_call_position_if_any_1', '2nd_call_position_if_any_2',
    '2nd_call_position_if_any_3', '2nd_call_position_if_any_4', '2nd_call_position_if_any_5',
    '2nd_call_position_if_any_6', '2nd_call_position_if_any_7', '2nd_call_position_if_any_8',
    '2nd_call_position_if_any_9', '2nd_call_position_if_any_10', 'gate_call_position_if_any_1',
    'gate_call_position_if_any_2', 'gate_call_position_if_any_3', 'gate_call_position_if_any_4',
    'gate_call_position_if_any_5', 'gate_call_position_if_any_6', 'gate_call_position_if_any_7',
    'gate_call_position_if_any_8', 'gate_call_position_if_any_9', 'gate_call_position_if_any_10',
    'stretch_position_if_any_1', 'stretch_position_if_any_2', 'stretch_position_if_any_3',
    'stretch_position_if_any_4', 'stretch_position_if_any_5', 'stretch_position_if_any_6',
    'stretch_position_if_any_7', 'stretch_position_if_any_8', 'stretch_position_if_any_9',
    'stretch_position_if_any_10', 'finish_position_1', 'finish_position_2', 'finish_position_3',
    'finish_position_4', 'finish_position_5', 'finish_position_6', 'finish_position_7',
    'finish_position_8', 'finish_position_9', 'finish_position_10', 'money_position_1',
    'money_position_2', 'money_position_3', 'money_position_4', 'money_position_5', 'money_position_6',
    'money_position_7', 'money_position_8', 'money_position_9', 'money_position_10',
    'start_call_btnlngths_ldr_margin_99_99_5_1', 'start_call_btnlngths_ldr_margin_99_99_5_2',
    'start_call_btnlngths_ldr_margin_99_99_5_3', 'start_call_btnlngths_ldr_margin_99_99_5_4',
    'start_call_btnlngths_ldr_margin_99_99_5_5', 'start_call_btnlngths_ldr_margin_99_99_5_6',
    'start_call_btnlngths_ldr_margin_99_99_5_7', 'start_call_btnlngths_ldr_margin_99_99_5_8',
    'start_call_btnlngths_ldr_margin_99_99_5_9', 'start_call_btnlngths_ldr_margin_99_99_5_10',
    'start_call_btnlngths_only_99_99_5_1', 'start_call_btnlngths_only_99_99_5_2',
    'start_call_btnlngths_only_99_99_5_3', 'start_call_btnlngths_only_99_99_5_4',
    'start_call_btnlngths_only_99_99_5_5', 'start_call_btnlngths_only_99_99_5_6',
    'start_call_btnlngths_only_99_99_5_7', 'start_call_btnlngths_only_99_99_5_8',
    'start_call_btnlngths_only_99_99_5_9', 'start_call_btnlngths_only_99_99_5_10',
    '1st_call_btnlngths_ldr_margin_1', '1st_call_btnlngths_ldr_margin_2', '1st_call_btnlngths_ldr_margin_3',
    '1st_call_btnlngths_ldr_margin_4', '1st_call_btnlngths_ldr_margin_5', '1st_call_btnlngths_ldr_margin_6',
    '1st_call_btnlngths_ldr_margin_7', '1st_call_btnlngths_ldr_margin_8', '1st_call_btnlngths_ldr_margin_9',
    '1st_call_btnlngths_ldr_margin_10', '1st_call_btnlngths_only_1', '1st_call_btnlngths_only_2',
    '1st_call_btnlngths_only_3', '1st_call_btnlngths_only_4', '1st_call_btnlngths_only_5',
    '1st_call_btnlngths_only_6', '1st_call_btnlngths_only_7', '1st_call_btnlngths_only_8',
    '1st_call_btnlngths_only_9', '1st_call_btnlngths_only_10', '2nd_call_btnlngths_ldr_margin_1',
    '2nd_call_btnlngths_ldr_margin_2', '2nd_call_btnlngths_ldr_margin_3', '2nd_call_btnlngths_ldr_margin_4',
    '2nd_call_btnlngths_ldr_margin_5', '2nd_call_btnlngths_ldr_margin_6', '2nd_call_btnlngths_ldr_margin_7',
    '2nd_call_btnlngths_ldr_margin_8', '2nd_call_btnlngths_ldr_margin_9', '2nd_call_btnlngths_ldr_margin_10',
    '2nd_call_btnlngths_only_1', '2nd_call_btnlngths_only_2', '2nd_call_btnlngths_only_3',
    '2nd_call_btnlngths_only_4', '2nd_call_btnlngths_only_5', '2nd_call_btnlngths_only_6',
    '2nd_call_btnlngths_only_7', '2nd_call_btnlngths_only_8', '2nd_call_btnlngths_only_9',
    '2nd_call_btnlngths_only_10', 'bris_race_shape_1st_call_1', 'bris_race_shape_1st_call_2',
    'bris_race_shape_1st_call_3', 'bris_race_shape_1st_call_4', 'bris_race_shape_1st_call_5',
    'bris_race_shape_1st_call_6', 'bris_race_shape_1st_call_7', 'bris_race_shape_1st_call_8',
    'bris_race_shape_1st_call_9', 'bris_race_shape_1st_call_10', 'stretch_btnlngths_ldr_margin_1',
    'stretch_btnlngths_ldr_margin_2', 'stretch_btnlngths_ldr_margin_3', 'stretch_btnlngths_ldr_margin_4',
    'stretch_btnlngths_ldr_margin_5', 'stretch_btnlngths_ldr_margin_6', 'stretch_btnlngths_ldr_margin_7',
    'stretch_btnlngths_ldr_margin_8', 'stretch_btnlngths_ldr_margin_9', 'stretch_btnlngths_ldr_margin_10',
    'stretch_btnlngths_only_1', 'stretch_btnlngths_only_2', 'stretch_btnlngths_only_3',
    'stretch_btnlngths_only_4', 'stretch_btnlngths_only_5', 'stretch_btnlngths_only_6',
    'stretch_btnlngths_only_7', 'stretch_btnlngths_only_8', 'stretch_btnlngths_only_9',
    'stretch_btnlngths_only_10', 'finish_btnlngths_wnrs_marginnumeric_99_99_5_1',
    'finish_btnlngths_wnrs_marginnumeric_99_99_5_2', 'finish_btnlngths_wnrs_marginnumeric_99_99_5_3',
    'finish_btnlngths_wnrs_marginnumeric_99_99_5_4', 'finish_btnlngths_wnrs_marginnumeric_99_99_5_5',
    'finish_btnlngths_wnrs_marginnumeric_99_99_5_6', 'finish_btnlngths_wnrs_marginnumeric_99_99_5_7',
    'finish_btnlngths_wnrs_marginnumeric_99_99_5_8', 'finish_btnlngths_wnrs_marginnumeric_99_99_5_9',
    'finish_btnlngths_wnrs_marginnumeric_99_99_5_10', 'finish_btnlngths_only_1',
    'finish_btnlngths_only_2', 'finish_btnlngths_only_3', 'finish_btnlngths_only_4',
    'finish_btnlngths_only_5', 'finish_btnlngths_only_6', 'finish_btnlngths_only_7',
    'finish_btnlngths_only_8', 'finish_btnlngths_only_9', 'finish_btnlngths_only_10',
    'bris_race_shape_2nd_call_1', 'bris_race_shape_2nd_call_2', 'bris_race_shape_2nd_call_3',
    'bris_race_shape_2nd_call_4', 'bris_race_shape_2nd_call_5', 'bris_race_shape_2nd_call_6',
    'bris_race_shape_2nd_call_7', 'bris_race_shape_2nd_call_8', 'bris_race_shape_2nd_call_9',
    'bris_race_shape_2nd_call_10', 'bris_2f_pace_fig_1', 'bris_2f_pace_fig_2', 'bris_2f_pace_fig_3',
    'bris_2f_pace_fig_4', 'bris_2f_pace_fig_5', 'bris_2f_pace_fig_6', 'bris_2f_pace_fig_7',
    'bris_2f_pace_fig_8', 'bris_2f_pace_fig_9', 'bris_2f_pace_fig_10', 'bris_4f_pace_fig_1',
    'bris_4f_pace_fig_2', 'bris_4f_pace_fig_3', 'bris_4f_pace_fig_4', 'bris_4f_pace_fig_5',
    'bris_4f_pace_fig_6', 'bris_4f_pace_fig_7', 'bris_4f_pace_fig_8', 'bris_4f_pace_fig_9',
    'bris_4f_pace_fig_10', 'bris_6f_pace_fig_1', 'bris_6f_pace_fig_2', 'bris_6f_pace_fig_3',
    'bris_6f_pace_fig_4', 'bris_6f_pace_fig_5', 'bris_6f_pace_fig_6', 'bris_6f_pace_fig_7',
    'bris_6f_pace_fig_8', 'bris_6f_pace_fig_9', 'bris_6f_pace_fig_10', 'bris_8f_pace_fig_1',
    'bris_8f_pace_fig_2', 'bris_8f_pace_fig_3', 'bris_8f_pace_fig_4', 'bris_8f_pace_fig_5',
    'bris_8f_pace_fig_6', 'bris_8f_pace_fig_7', 'bris_8f_pace_fig_8', 'bris_8f_pace_fig_9',
    'bris_8f_pace_fig_10', 'bris_10f_pace_fig_1', 'bris_10f_pace_fig_2', 'bris_10f_pace_fig_3',
    'bris_10f_pace_fig_4', 'bris_10f_pace_fig_5', 'bris_10f_pace_fig_6', 'bris_10f_pace_fig_7',
    'bris_10f_pace_fig_8', 'bris_10f_pace_fig_9', 'bris_10f_pace_fig_10', 'bris_late_pace_fig_1',
    'bris_late_pace_fig_2', 'bris_late_pace_fig_3', 'bris_late_pace_fig_4', 'bris_late_pace_fig_5',
    'bris_late_pace_fig_6', 'bris_late_pace_fig_7', 'bris_late_pace_fig_8', 'bris_late_pace_fig_9',
    'bris_late_pace_fig_10', 'bris_speed_rating_1', 'bris_speed_rating_2', 'bris_speed_rating_3',
    'bris_speed_rating_4', 'bris_speed_rating_5', 'bris_speed_rating_6', 'bris_speed_rating_7',
    'bris_speed_rating_8', 'bris_speed_rating_9', 'bris_speed_rating_10', 'speed_rating_1',
    'speed_rating_2', 'speed_rating_3', 'speed_rating_4', 'speed_rating_5', 'speed_rating_6',
    'speed_rating_7', 'speed_rating_8', 'speed_rating_9', 'speed_rating_10', 'track_variant_1',
    'track_variant_2', 'track_variant_3', 'track_variant_4', 'track_variant_5', 'track_variant_6',
    'track_variant_7', 'track_variant_8', 'track_variant_9', 'track_variant_10', '2f_fraction_if_any_1',
    '2f_fraction_if_any_2', '2f_fraction_if_any_3', '2f_fraction_if_any_4', '2f_fraction_if_any_5',
    '2f_fraction_if_any_6', '2f_fraction_if_any_7', '2f_fraction_if_any_8', '2f_fraction_if_any_9',
    '2f_fraction_if_any_10', '3f_fraction_if_any_1', '3f_fraction_if_any_2', '3f_fraction_if_any_3',
    '3f_fraction_if_any_4', '3f_fraction_if_any_5', '3f_fraction_if_any_6', '3f_fraction_if_any_7',
    '3f_fraction_if_any_8', '3f_fraction_if_any_9', '3f_fraction_if_any_10', '4f_fraction_if_any_1',
    '4f_fraction_if_any_2', '4f_fraction_if_any_3', '4f_fraction_if_any_4', '4f_fraction_if_any_5',
    '4f_fraction_if_any_6', '4f_fraction_if_any_7', '4f_fraction_if_any_8', '4f_fraction_if_any_9',
    '4f_fraction_if_any_10', '5f_fraction_if_any_1', '5f_fraction_if_any_2', '5f_fraction_if_any_3',
    '5f_fraction_if_any_4', '5f_fraction_if_any_5', '5f_fraction_if_any_6', '5f_fraction_if_any_7',
    '5f_fraction_if_any_8', '5f_fraction_if_any_9', '5f_fraction_if_any_10', '6f_fraction_if_any_1',
    '6f_fraction_if_any_2', '6f_fraction_if_any_3', '6f_fraction_if_any_4', '6f_fraction_if_any_5',
    '6f_fraction_if_any_6', '6f_fraction_if_any_7', '6f_fraction_if_any_8', '6f_fraction_if_any_9',
    '6f_fraction_if_any_10', '7f_fraction_if_any_1', '7f_fraction_if_any_2', '7f_fraction_if_any_3',
    '7f_fraction_if_any_4', '7f_fraction_if_any_5', '7f_fraction_if_any_6', '7f_fraction_if_any_7',
    '7f_fraction_if_any_8', '7f_fraction_if_any_9', '7f_fraction_if_any_10', '8f_fraction_if_any_1',
    '8f_fraction_if_any_2', '8f_fraction_if_any_3', '8f_fraction_if_any_4', '8f_fraction_if_any_5',
    '8f_fraction_if_any_6', '8f_fraction_if_any_7', '8f_fraction_if_any_8', '8f_fraction_if_any_9',
    '8f_fraction_if_any_10', '10f_fraction_if_any_1', '10f_fraction_if_any_2', '10f_fraction_if_any_3',
    '10f_fraction_if_any_4', '10f_fraction_if_any_5', '10f_fraction_if_any_6', '10f_fraction_if_any_7',
    '10f_fraction_if_any_8', '10f_fraction_if_any_9', '10f_fraction_if_any_10', '12f_fraction_if_any_1',
    '12f_fraction_if_any_2', '12f_fraction_if_any_3', '12f_fraction_if_any_4', '12f_fraction_if_any_5',
    '12f_fraction_if_any_6', '12f_fraction_if_any_7', '12f_fraction_if_any_8', '12f_fraction_if_any_9',
    '12f_fraction_if_any_10', '14f_fraction_if_any_1', '14f_fraction_if_any_2', '14f_fraction_if_any_3',
    '14f_fraction_if_any_4', '14f_fraction_if_any_5', '14f_fraction_if_any_6', '14f_fraction_if_any_7',
    '14f_fraction_if_any_8', '14f_fraction_if_any_9', '14f_fraction_if_any_10', '16f_fraction_if_any_1',
    '16f_fraction_if_any_2', '16f_fraction_if_any_3', '16f_fraction_if_any_4', '16f_fraction_if_any_5',
    '16f_fraction_if_any_6', '16f_fraction_if_any_7', '16f_fraction_if_any_8', '16f_fraction_if_any_9',
    '16f_fraction_if_any_10', 'fraction_1_1', 'fraction_1_2', 'fraction_1_3', 'fraction_1_4',
    'fraction_1_5', 'fraction_1_6', 'fraction_1_7', 'fraction_1_8', 'fraction_1_9', 'fraction_1_10',
    '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '2_10', '3_1', '3_2', '3_3', '3_4',
    '3_5', '3_6', '3_7', '3_8', '3_9', '3_10', 'final_time_1', 'final_time_2', 'final_time_3',
    'final_time_4', 'final_time_5', 'final_time_6', 'final_time_7', 'final_time_8', 'final_time_9',
    'final_time_10', 'claimed_code_1', 'claimed_code_2', 'claimed_code_3', 'claimed_code_4',
    'claimed_code_5', 'claimed_code_6', 'claimed_code_7', 'claimed_code_8', 'claimed_code_9',
    'claimed_code_10', 'trainer_when_available_1', 'trainer_when_available_2', 'trainer_when_available_3',
    'trainer_when_available_4', 'trainer_when_available_5', 'trainer_when_available_6',
    'trainer_when_available_7', 'trainer_when_available_8', 'trainer_when_available_9',
    'trainer_when_available_10', 'jockey_1', 'jockey_2', 'jockey_3', 'jockey_4', 'jockey_5', 'jockey_6',
    'jockey_7', 'jockey_8', 'jockey_9', 'jockey_10', 'apprentice_wt_allow_if_any_1',
    'apprentice_wt_allow_if_any_2', 'apprentice_wt_allow_if_any_3', 'apprentice_wt_allow_if_any_4',
    'apprentice_wt_allow_if_any_5', 'apprentice_wt_allow_if_any_6', 'apprentice_wt_allow_if_any_7',
    'apprentice_wt_allow_if_any_8', 'apprentice_wt_allow_if_any_9', 'apprentice_wt_allow_if_any_10',
    'race_type_1', 'race_type_2', 'race_type_3', 'race_type_4', 'race_type_5', 'race_type_6',
    'race_type_7', 'race_type_8', 'race_type_9', 'race_type_10', 'age_and_sex_restrictions_1',
    'age_and_sex_restrictions_2', 'age_and_sex_restrictions_3', 'age_and_sex_restrictions_4',
    'age_and_sex_restrictions_5', 'age_and_sex_restrictions_6', 'age_and_sex_restrictions_7',
    'age_and_sex_restrictions_8', 'age_and_sex_restrictions_9', 'age_and_sex_restrictions_10',
    'statebred_flag_1', 'statebred_flag_2', 'statebred_flag_3', 'statebred_flag_4', 'statebred_flag_5',
    'statebred_flag_6', 'statebred_flag_7', 'statebred_flag_8', 'statebred_flag_9', 'statebred_flag_10',
    'restricted_qualifier_flag_1', 'restricted_qualifier_flag_2', 'restricted_qualifier_flag_3',
    'restricted_qualifier_flag_4', 'restricted_qualifier_flag_5', 'restricted_qualifier_flag_6',
    'restricted_qualifier_flag_7', 'restricted_qualifier_flag_8', 'restricted_qualifier_flag_9',
    'restricted_qualifier_flag_10', 'favorite_indicator_1', 'favorite_indicator_2',
    'favorite_indicator_3', 'favorite_indicator_4', 'favorite_indicator_5', 'favorite_indicator_6',
    'favorite_indicator_7', 'favorite_indicator_8', 'favorite_indicator_9', 'favorite_indicator_10',
    'front_bandages_indicator_1', 'front_bandages_indicator_2', 'front_bandages_indicator_3',
    'front_bandages_indicator_4', 'front_bandages_indicator_5', 'front_bandages_indicator_6',
    'front_bandages_indicator_7', 'front_bandages_indicator_8', 'front_bandages_indicator_9',
    'front_bandages_indicator_10', 'low_claiming_price_of_race_1', 'low_claiming_price_of_race_2',
    'low_claiming_price_of_race_3', 'low_claiming_price_of_race_4', 'low_claiming_price_of_race_5',
    'low_claiming_price_of_race_6', 'low_claiming_price_of_race_7', 'low_claiming_price_of_race_8',
    'low_claiming_price_of_race_9', 'low_claiming_price_of_race_10', 'high_claiming_price_of_race_1',
    'high_claiming_price_of_race_2', 'high_claiming_price_of_race_3', 'high_claiming_price_of_race_4',
    'high_claiming_price_of_race_5', 'high_claiming_price_of_race_6', 'high_claiming_price_of_race_7',
    'high_claiming_price_of_race_8', 'high_claiming_price_of_race_9', 'high_claiming_price_of_race_10',
    'bar_shoe_1', 'bar_shoe_2', 'bar_shoe_3', 'bar_shoe_4', 'bar_shoe_5', 'bar_shoe_6', 'bar_shoe_7',
    'bar_shoe_8', 'bar_shoe_9', 'bar_shoe_10', 'code_for_prior_10_starts_1_s_nasal_strip_1',
    'code_for_prior_10_starts_1_s_nasal_strip_2', 'code_for_prior_10_starts_1_s_nasal_strip_3',
    'code_for_prior_10_starts_1_s_nasal_strip_4', 'code_for_prior_10_starts_1_s_nasal_strip_5',
    'code_for_prior_10_starts_1_s_nasal_strip_6', 'code_for_prior_10_starts_1_s_nasal_strip_7',
    'code_for_prior_10_starts_1_s_nasal_strip_8', 'code_for_prior_10_starts_1_s_nasal_strip_9',
    'code_for_prior_10_starts_1_s_nasal_strip_10', 'extended_start_comment_1', 'extended_start_comment_2',
    'extended_start_comment_3', 'extended_start_comment_4', 'extended_start_comment_5',
    'extended_start_comment_6', 'extended_start_comment_7', 'extended_start_comment_8',
    'extended_start_comment_9', 'extended_start_comment_10', 'sealed_track_indicator_1',
    'sealed_track_indicator_2', 'sealed_track_indicator_3', 'sealed_track_indicator_4',
    'sealed_track_indicator_5', 'sealed_track_indicator_6', 'sealed_track_indicator_7',
    'sealed_track_indicator_8', 'sealed_track_indicator_9', 'sealed_track_indicator_10',
    'prev_all_weather_surface_flagcharacter_1', 'prev_all_weather_surface_flagcharacter_2',
    'prev_all_weather_surface_flagcharacter_3', 'prev_all_weather_surface_flagcharacter_4',
    'prev_all_weather_surface_flagcharacter_5', 'prev_all_weather_surface_flagcharacter_6',
    'prev_all_weather_surface_flagcharacter_7', 'prev_all_weather_surface_flagcharacter_8',
    'prev_all_weather_surface_flagcharacter_9', 'prev_all_weather_surface_flagcharacter_10',
    'equibase_abbrev_race_conditions_character_17_17_1', 'equibase_abbrev_race_conditions_character_17_17_2',
    'equibase_abbrev_race_conditions_character_17_17_3', 'equibase_abbrev_race_conditions_character_17_17_4',
    'equibase_abbrev_race_conditions_character_17_17_5', 'equibase_abbrev_race_conditions_character_17_17_6',
    'equibase_abbrev_race_conditions_character_17_17_7', 'equibase_abbrev_race_conditions_character_17_17_8',
    'equibase_abbrev_race_conditions_character_17_17_9', 'equibase_abbrev_race_conditions_character_17_17_10',
    'blank_fields_reserved_for_possible_future_expansion_1', 'blank_fields_reserved_for_possible_future_expansion_2',
    'blank_fields_reserved_for_possible_future_expansion_3'
]

record_groups = {
    "distance":     ("starts_pos_65", "wins_pos_66", "places_pos_67", "shows_pos_68", "earnings_pos_69"),
    "track":        ("starts_pos_70", "wins_pos_71", "places_pos_72", "shows_pos_73", "earnings_pos_74"),
    "turf":         ("starts_pos_75", "wins_pos_76", "places_pos_77", "shows_pos_78", "earnings_pos_79"),
    "wet":          ("starts_pos_80", "wins_pos_81", "places_pos_82", "shows_pos_83", "earnings_pos_84"),
    "current_year": ("starts_pos_86", "wins_pos_87", "places_pos_88", "shows_pos_89", "earnings_pos_90"),
    "previous_year":("starts_pos_92", "wins_pos_93", "places_pos_94", "shows_pos_95", "earnings_pos_96"),
    "lifetime":     ("starts_pos_97", "wins_pos_98", "places_pos_99", "shows_pos_100", "earnings_pos_101"),
}

# --- Main Function (New) ---
def main():
    """
    Main function to process current race info.
    This function will be called by run_pipeline.py.
    """
    logger = logging.getLogger(__name__) # Get logger instance
    logger.info(f"--- Creating Current Race Info File ({pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    # 1. Load the full wide-format data
    if not WIDE_DATA_FILE_PATH.exists():
        logger.error(f"Error: Input Parquet file not found at {WIDE_DATA_FILE_PATH}")
        # Consider raising an error or returning a status
        return
    try:
        logger.info(f"Loading wide format data from: {WIDE_DATA_FILE_PATH}")
        wide_df = pd.read_parquet(WIDE_DATA_FILE_PATH, engine='pyarrow')
        logger.info(f"Loaded wide data with shape: {wide_df.shape}")
        original_cols = wide_df.columns.tolist()
    except Exception as e:
        logger.error(f"Error loading Parquet file {WIDE_DATA_FILE_PATH}: {e}", exc_info=True)
        return

    # 2. Identify columns that actually exist in the DataFrame and are in the drop list
    cols_to_actually_drop = [col for col in COLUMNS_TO_DROP if col in original_cols]
    if len(cols_to_actually_drop) < len(COLUMNS_TO_DROP):
         missing_from_df = [col for col in COLUMNS_TO_DROP if col not in original_cols]
         logger.warning(f"\nWarning: {len(missing_from_df)} columns listed in COLUMNS_TO_DROP were not found in the DataFrame.")

    # 3. Drop the identified columns
    logger.info(f"\nDropping {len(cols_to_actually_drop)} wide workout/past performance columns...")
    current_info_df = wide_df.drop(columns=cols_to_actually_drop, errors='ignore')

    # --- Defensive numeric conversion for record columns ---
    cols_to_convert_to_numeric = []
    for _, group_cols in record_groups.items():
        cols_to_convert_to_numeric.extend(group_cols)

    for col in set(cols_to_convert_to_numeric):
        if col in current_info_df.columns:
            current_info_df[col] = pd.to_numeric(current_info_df[col], errors='coerce')
        else:
            logger.warning(f"Warning: Column {col} for record calculation not found in DataFrame.")

    # --- Build list of metric names ---
    record_group_names = [
        "distance", "track", "turf", "wet",
        "current_year", "previous_year", "lifetime"
    ]
    # metrics = [] # This variable is defined but not used later in this block
    for name_rg in record_group_names: # Renamed 'name' to 'name_rg' to avoid conflict
        # Check if all necessary columns for the current record group exist
        group_cols_exist = True
        if name_rg in record_groups:
            for col_check in record_groups[name_rg]:
                if col_check not in current_info_df.columns:
                    logger.warning(f"Missing column '{col_check}' for record group '{name_rg}'. Skipping this group.")
                    group_cols_exist = False
                    break
        else:
            logger.warning(f"Record group '{name_rg}' not defined in record_groups. Skipping.")
            group_cols_exist = False

        if group_cols_exist:
            col_start, col_win, col_place, col_show, col_earn = record_groups[name_rg]
            starts = current_info_df[col_start].replace({0: np.nan})
            wins   = current_info_df[col_win]
            # Assuming 'itm' was intended to be wins + place + show based on typical racing stats for "in the money"
            # Original code had: itm = current_info_df[col_place] + current_info_df[col_show]
            # Let's assume ITM = Win, Place, Show counts
            itm    = current_info_df[col_win] + current_info_df[col_place] + current_info_df[col_show]
            earnings = current_info_df[col_earn]

            current_info_df[f"{name_rg}_win_pct"] = wins / starts
            current_info_df[f"{name_rg}_itm_pct"] = itm  / starts # ITM = In The Money (1st, 2nd, or 3rd)
            current_info_df[f"{name_rg}_earnings_per_start"] = earnings / starts

    logger.info(f"DataFrame shape after dropping columns: {current_info_df.shape}")
    dropped_count = len(original_cols) - len(current_info_df.columns)
    logger.info(f"Number of columns actually dropped: {dropped_count}")

    #4a. Add a furlongs column
    if 'distance_in_yards' in current_info_df.columns:
        current_info_df['distance_in_yards'] = pd.to_numeric(
            current_info_df['distance_in_yards'], errors='coerce'
        )
    else:
        logger.warning("Warning: 'distance_in_yards' not found; cannot calculate furlongs.")
        current_info_df['distance_in_yards'] = np.nan # Create the column to avoid error in next line

    current_info_df['furlongs'] = current_info_df['distance_in_yards'] / 220

    # 4b. Save the resulting DataFrame
    logger.info(f"\nSaving current race info data to: {CURRENT_INFO_FILE_PATH}")
    try:
        CURRENT_INFO_FILE_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        current_info_df.to_parquet(CURRENT_INFO_FILE_PATH, index=False, engine='pyarrow')
        logger.info("Save complete.")
        logger.info("\nOutput DataFrame Info:")
        # current_info_df.info(verbose=False, show_counts=True) # .info() prints to stdout
    except ImportError:
        logger.error("\nError: 'pyarrow' library not found. Cannot save to Parquet.")
        logger.error("Please install it: pip install pyarrow")
    except Exception as e:
        logger.error(f"\nError saving final data to Parquet file {CURRENT_INFO_FILE_PATH}: {e}", exc_info=True)

    logger.info(f"\n--- Script Finished ({pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

# --- Main Execution (for direct script run) ---
if __name__ == "__main__":
    # Setup basic logging if this script is run directly
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)] # Use sys for direct runs
        )
    main()