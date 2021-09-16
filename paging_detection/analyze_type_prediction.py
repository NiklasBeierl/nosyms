from collections import defaultdict, Counter
from functools import cache
import json
from typing import Dict

import pandas as pd

from paging_detection import Snapshot, PageTypes, PagingStructure

_PAGE_TYPES_ORDERED = tuple(PageTypes)


class CachedMapsToData:
    def __init__(self, pages: Dict[int, PagingStructure]):
        self.pages = pages

    @cache
    def __call__(self, page: int, page_type: PageTypes) -> bool:
        """
        Determine if a page under a certain designation ends up mapping to any data pages.
        """

        page = self.pages[page]

        assert page_type in page.designations

        if page_type == PageTypes.PT and page.entries:
            return True

        for entry in page.entries.values():
            if entry.target_is_data(page_type):
                return True
            else:
                return self(entry.target, _PAGE_TYPES_ORDERED[_PAGE_TYPES_ORDERED.index(page_type) + 1])
        return False


def calculate_errors(pages_truth, pages_predicted):
    maps_to_data = CachedMapsToData(pages_truth)
    true_positives = defaultdict(lambda: 0)
    false_positives = defaultdict(lambda: 0)
    true_negatives = defaultdict(lambda: 0)
    # Pages not mapping to any data are not counted here
    false_negatives = defaultdict(lambda: 0)
    # Considering false negatives where the real page does not map to any data
    false_negatives_with_empty = defaultdict(lambda: 0)

    for offset, pred in pages_predicted.items():
        true_designations = pages_truth[offset].designations if offset in pages_truth else set()

        for page_type in true_designations & pred.designations:
            true_positives[page_type] += 1
        for page_type in set(PageTypes) - (true_designations | pred.designations):
            true_negatives[page_type] += 1
        for page_type in pred.designations - true_designations:
            false_positives[page_type] += 1
        for page_type in true_designations - pred.designations:
            false_negatives_with_empty[page_type] += 1
            if maps_to_data(offset, page_type):
                false_negatives[page_type] += 1
    return true_positives, false_positives, true_negatives, false_negatives, false_negatives_with_empty


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="Snapshot JSON with predicted designations.", type=pathlib.Path)
    parser.add_argument("truths", help="Snapshot JSON with true paging structures.", type=pathlib.Path)
    args = parser.parse_args()

    print("Loading page data.")
    with open(args.predictions) as f:
        predicted = Snapshot.validate(json.load(f))

    with open(args.truths) as f:
        truth = Snapshot.validate(json.load(f))

    print("Filtering out of bound entries.")
    for page in truth.pages.values():
        page.entries = {offset: entry for offset, entry in page.entries.items() if entry.target < truth.size}

    print("Counting errors.")
    tp, fp, tn, fn, fnwe = calculate_errors(truth.pages, predicted.pages)

    truth_counts = Counter((page_type for page in truth.pages.values() for page_type in page.designations))

    summary = [(pt, tp[pt], fp[pt], tn[pt], fn[pt], fnwe[pt]) for pt in PageTypes]
    summary_df = pd.DataFrame(summary, columns=["Type", "TP", "FP", "TN", "FN", "FN (with empty)"]).set_index("Type")
    summary_df["true counts"] = pd.Series(truth_counts)

    total = len(predicted.pages)
    summary_df["accuracy"] = (summary_df["TP"] + summary_df["TN"]) / total
    summary_df["recall"] = summary_df["TP"] / summary_df["true counts"]
    summary_df["precision"] = summary_df["TP"] / (summary_df["TP"] + summary_df["FP"])
    print(summary_df.to_string())

print("Done")
