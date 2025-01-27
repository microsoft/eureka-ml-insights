# This file was authored by BenchAgents authors and is being reused under the MIT license.
# All code in this file is directly copied from the original source repository.
# https://github.com/microsoft/benchagents

import ast
import json
import re
import numpy as np
from datetime import datetime, timedelta

import pandas as pd

from eureka_ml_insights.metrics.metrics_base import CompositeMetric

# Helper functions
def check_time_slot_format(solution):
    pattern = r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) ([0-9]|[01]\d|2[0-3]):[0-5]\d-([0-9]|[01]\d|2[0-3]):[0-5]\d$"
    return bool(re.match(pattern, solution))

def generate_time_slots(start_time, end_time, granularity):
    granularity=5
    slots = []
    current_time = start_time
    while current_time + timedelta(minutes=granularity) <= end_time:
        slots.append((current_time, current_time + timedelta(minutes=granularity)))
        current_time += timedelta(minutes=granularity)
    return slots

def parse_time_block(time_block):
    start_str, end_str = time_block.split('-')
    start_time = datetime.strptime(start_str, "%H:%M")
    end_time = datetime.strptime(end_str, "%H:%M")
    return start_time, end_time

def filter_slots_by_duration(time_slots, duration):
    filtered_slots = []
    for i in range(len(time_slots)):
        accumulated_duration = timedelta()
        for j in range(i, len(time_slots)):
            accumulated_duration += time_slots[j][1] - time_slots[j][0]
            if accumulated_duration >= timedelta(minutes=duration):
                filtered_slots.append((time_slots[i][0], time_slots[j][1]))
                break
    return filtered_slots

def filter_slots_by_constraints(time_slots, constraints, day):
    filtered_slots = []
    for slot in time_slots:
        start_time, end_time = slot
        if constraints['no_meetings_before']:
            nb = int(constraints['no_meetings_before'])
            no_meetings_before = datetime.strptime(f"{nb}:00", "%H:%M")
            if start_time < no_meetings_before:
                continue
        if constraints['no_meetings_after']:
            na = int(constraints['no_meetings_after'])
            no_meetings_after = datetime.strptime(f"{na}:00", "%H:%M")
            if end_time >= no_meetings_after:
                continue
        if constraints['no_meetings_on_weekends'] and day in ['Saturday', 'Sunday']:
            continue
        if constraints['no_meetings_during_specific_times']:
            no_meetings_start, no_meetings_end = parse_time_block(constraints['no_meetings_during_specific_times'])
            if (start_time < no_meetings_end and end_time > no_meetings_start):
                continue
        filtered_slots.append(slot)
    return filtered_slots

class BACalendarMetric(CompositeMetric):
    """
    Composite metric for evaluating if a response for each criteria.

    This metric evaluates if a given response follows the provided constraints.
    """

    def __init__(self):
        super().__init__()
        self.no_solution_response = "No common time slot available"

    def __evaluate__(self, row):
        results = {}
        results.update(self.run_programmatic_tests(row))
        return results

    def run_programmatic_tests(self, instance):
        result = {}
        solution = instance['model_output']
        solution = solution.strip('"').strip('`').strip('\n')
        if check_time_slot_format(solution):
            result['format_programmatic'] = 1
        result.update(self.check_availability_programmatic(instance, solution))
        result.update(self.check_meeting_duration_programmatic(instance, solution))
        result.update(self.check_buffer_time_programmatic(instance, solution))
        result.update(self.check_no_weekends_programmatic(instance, solution))
        result.update(self.check_time_restrictions_programmatic(instance, solution))
        result.update(self.check_specific_times_programmatic(instance, solution))
        result.update(self.check_priority_programmatic(instance, solution))
        all_correct = 1
        passed_constraints = []
        for key, value in result.items():
            if value == 0:
                all_correct = 0
            if value is not None and value != 'NA' and pd.notna(value) and isinstance(value, int):
                passed_constraints.append(value)
        result['all_correct'] = all_correct
        result['fraction_passed'] = np.mean(passed_constraints)
        return result

    def is_formatted(self, solution):
        run_tests=True
        if solution == self.no_solution_response:
            run_tests=False
        if not check_time_slot_format(solution):
            run_tests=False
        return run_tests

    def check_availability_programmatic(self, instance, solution):
        if not instance['constraints'].get('availability', True):
            # result = {'availability_programmatic_check': 'NA'}
            result = {'availability_programmatic_check': None}
            return result
        
        if not self.is_formatted(solution):
            result = {'availability_programmatic_check': 0}
            return result

        day, time_range = solution.split()
        start_time, end_time = parse_time_block(time_range)
        all_available = 1
        availability = json.loads(instance['metadata']['availability'].replace("'", '"'))
        for participant, schedule in availability.items():
            if day not in schedule:
                all_available = 0
                break
            available_blocks = schedule[day]
            available = False
            for block in available_blocks:
                block_start, block_end = parse_time_block(block)
                if block_start <= start_time and block_end >= end_time:
                    available = True
                    break
            if not available:
                all_available = 0
                break

        return {'availability_programmatic_check': all_available}

    def check_meeting_duration_programmatic(self, instance, solution):
        if not instance['constraints'].get('meeting_duration', True):
            # result = {'meeting_duration_programmatic_check': 'NA'}
            result = {'meeting_duration_programmatic_check': None}
            return result
        
        if not self.is_formatted(solution):
            result = {'meeting_duration_programmatic_check': 0}
            return result

        _, time_range = solution.split()
        start_time, end_time = parse_time_block(time_range)
        meeting_duration = (end_time - start_time).total_seconds() / 60
        expected_duration = instance['constraints']['meeting_duration']

        return {'meeting_duration_programmatic_check': int(meeting_duration == expected_duration)}


    def check_buffer_time_programmatic(self, instance, solution):
        buffer_time = instance['constraints'].get('buffer_time_before_and_after_meeting', True)
        if buffer_time is None or not buffer_time:
            # result = {'buffer_time_programmatic_check': 'NA'}
            result = {'buffer_time_programmatic_check': None}
            return result
        
        if not self.is_formatted(solution):
            result = {'buffer_time_programmatic_check': 0}
            return result

        buffer_time = instance['constraints']['buffer_time_before_and_after_meeting']
        day, time_range = solution.split()
        start_time, end_time = parse_time_block(time_range)
        buffer_start_time = start_time - timedelta(minutes=buffer_time)
        buffer_end_time = end_time + timedelta(minutes=buffer_time)
        all_buffer_respected = 1

        availability = json.loads(instance['metadata']['availability'].replace("'", '"'))
        for participant, schedule in availability.items():
            if day not in schedule:
                all_buffer_respected = 0
                break
            available_blocks = schedule[day]
            buffer_respected = False
            for block in available_blocks:
                block_start, block_end = parse_time_block(block)
                if block_start <= buffer_start_time and block_end >= buffer_end_time:
                    buffer_respected = True
                    break
            if not buffer_respected:
                all_buffer_respected = 0
                break
        return {'buffer_time_programmatic_check': all_buffer_respected}

    def check_no_weekends_programmatic(self, instance, solution):
        if not instance['constraints'].get('no_meetings_on_weekends', True):
            # return {'no_weekends_programmatic_check': 'NA'}
            return {'no_weekends_programmatic_check': None}
        
        if not self.is_formatted(solution):
            return {'no_weekends_programmatic_check': 0}

        day, _ = solution.split()
        day_of_week = datetime.strptime(day, '%A').weekday()
        no_weekends = day_of_week < 5
        return {'no_weekends_programmatic_check': int(no_weekends)}

    def check_time_restrictions_programmatic(self, instance, solution):
        if not instance['constraints'].get('no_meetings_before', True) and not instance['constraints'].get('no_meetings_after', True):
            # return {'time_restrictions_programmatic_check': 'NA'}
            return {'time_restrictions_programmatic_check': None}
        
        if not self.is_formatted(solution):
            return {'time_restrictions_programmatic_check': 0}

        _, time_range = solution.split()
        start_time, end_time = parse_time_block(time_range)

        no_meetings_before = instance['constraints'].get('no_meetings_before')
        no_meetings_after = instance['constraints'].get('no_meetings_after')

        if no_meetings_before:
            nb = int(no_meetings_before)
            no_meetings_before = datetime.strptime(f"{nb}:00", "%H:%M")
            if start_time < no_meetings_before:
                return {'time_restrictions_programmatic_check': 0}

        if no_meetings_after:
            na = int(no_meetings_after)
            no_meetings_after = datetime.strptime(f"{na}:00", '%H:%M')
            if end_time > no_meetings_after:
                return {'time_restrictions_programmatic_check': 0}
        return {'time_restrictions_programmatic_check': 1}

    def check_priority_programmatic(self, instance, solution):
        if not instance['constraints'].get('high_priority_meeting', False):
            # return {'priority_programmatic_check': 'NA'}
            return {'priority_programmatic_check': None}
        
        if not self.is_formatted(solution):
            return {'priority_programmatic_check': 0}
        
        metadata = instance['metadata']
        result = False
        params = instance['params']
        constraints = instance['constraints']
        if constraints['buffer_time_before_and_after_meeting']:
            buffer_time = constraints['buffer_time_before_and_after_meeting']
        else:
            buffer_time = 0
        for day in params['days_of_week']: # TODO: revisit this post data release to ensure consistency
            common_time_slots = None
            availability = json.loads(metadata['availability'].replace("'", '"'))
            for participant, schedule in availability.items():
                if day in schedule:
                    participant_time_slots = []
                    for time_slot in schedule[day]:
                        start_time, end_time = parse_time_block(time_slot)
                        time_slots = generate_time_slots(start_time, end_time, params['granularity'])
                        time_slots = filter_slots_by_duration(time_slots, constraints['meeting_duration'] + 2 * buffer_time)
                        time_slots = filter_slots_by_constraints(time_slots, constraints, day=day)
                        participant_time_slots.extend(time_slots)
                    if common_time_slots is None:
                        common_time_slots = set(participant_time_slots)
                    else:
                        common_time_slots = common_time_slots.intersection(participant_time_slots)
            if common_time_slots:
                first_available_slot = sorted(list(common_time_slots))[0]
                first_available_start = (first_available_slot[0]+timedelta(minutes=buffer_time)).strftime('%H:%M')
                first_available_end = (first_available_slot[1]-timedelta(minutes=buffer_time)).strftime('%H:%M')
                result = solution == f"{day} {first_available_start}-{first_available_end}"
        return {'priority_programmatic_check': int(result)}

    def check_specific_times_programmatic(self, instance, solution):
        if not instance['constraints'].get('no_meetings_during_specific_times', True):
            # return {'specific_times_programmatic_check': 'NA'}
            return {'specific_times_programmatic_check': None}
        
        if not self.is_formatted(solution):
            return {'specific_times_programmatic_check': 0}

        restricted_times = instance['constraints']['no_meetings_during_specific_times']
        restricted_start, restricted_end = parse_time_block(restricted_times)
        day, time_range = solution.split()
        start_time, end_time = parse_time_block(time_range)

        if (start_time < restricted_end and end_time > restricted_start):
            result = 0
        else:
            result = 1
        return {'specific_times_programmatic_check': result}
