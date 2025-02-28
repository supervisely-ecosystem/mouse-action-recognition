from make_clips import merge_overlapping_ranges

def test_merge_overlapping_ranges():
    ranges = [[0, 10], [5, 15], [20, 30], [25, 35]]
    result = merge_overlapping_ranges(ranges)
    assert result == [[0, 15], [20, 35]]

    ranges = [[0, 10], [15, 20], [25, 30]]
    result = merge_overlapping_ranges(ranges)
    assert result == [[0, 10], [15, 20], [25, 30]]

    ranges = [[0, 10], [15, 20], [25, 50], [35, 40]]
    result = merge_overlapping_ranges(ranges)
    assert result == [[0, 10], [15, 20], [25, 50]]

# test_merge_overlapping_ranges()


from pathlib import Path

p = Path("MP_TRAIN_3")
paths = list(p.rglob("*.MP4"))
paths += list(p.rglob("*.mp4"))
names = [path.name for path in paths]
f = [path for path in paths if path.stem in ['GL020636', 'GL010636', 'GL010625']]
print(names[0])
from collections import Counter
counter = Counter(names)
print(counter.most_common(5))