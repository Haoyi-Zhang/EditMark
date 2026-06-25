from __future__ import annotations

import json
from typing import Any

from .language_support import normalize_language_name


def json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def literal(language: str, value: Any) -> str:
    normalized = normalize_language_name(language)
    if normalized == "python":
        return repr(value)
    if normalized == "javascript":
        return json.dumps(value, ensure_ascii=False)
    if normalized == "cpp":
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return json_string(value)
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            return "{" + ", ".join(str(item) for item in value) + "}"
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return "{" + ", ".join(json_string(item) for item in value) + "}"
        raise TypeError(f"unsupported cpp literal: {value!r}")
    if normalized == "java":
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return json_string(value)
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            return "Arrays.asList(" + ", ".join(str(item) for item in value) + ")"
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return "Arrays.asList(" + ", ".join(json_string(item) for item in value) + ")"
        raise TypeError(f"unsupported java literal: {value!r}")
    if normalized == "go":
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return json_string(value)
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            return "[]int{" + ", ".join(str(item) for item in value) + "}"
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return "[]string{" + ", ".join(json_string(item) for item in value) + "}"
        raise TypeError(f"unsupported go literal: {value!r}")
    raise KeyError(language)


def reference_tests(language: str, entry_point: str, cases: list[dict[str, Any]]) -> tuple[str, ...]:
    normalized = normalize_language_name(language)
    if normalized == "python":
        return tuple(
            f"assert {entry_point}({', '.join(literal('python', arg) for arg in case['args'])}) == {literal('python', case['expected'])}"
            for case in cases
        )
    if normalized == "cpp":
        lines = ["#include <stdexcept>", "int main() {"]
        for index, case in enumerate(cases):
            args = ", ".join(literal("cpp", arg) for arg in case["args"])
            expected = literal("cpp", case["expected"])
            lines.extend(
                [
                    f"    auto actual_{index} = {entry_point}({args});",
                    f"    auto expected_{index} = {expected};",
                    f"    if (!(actual_{index} == expected_{index})) throw std::runtime_error(\"case_{index}\");",
                ]
            )
        lines.extend(["    return 0;", "}"])
        return ("\n".join(lines),)
    if normalized == "java":
        lines = ["import java.util.*;", "public class Main {", "    public static void main(String[] args) throws Exception {"]
        for index, case in enumerate(cases):
            args = ", ".join(literal("java", arg) for arg in case["args"])
            expected = literal("java", case["expected"])
            lines.extend(
                [
                    f"        Object actual_{index} = Solution.{entry_point}({args});",
                    f"        Object expected_{index} = {expected};",
                    f"        if (!Objects.equals(actual_{index}, expected_{index})) throw new RuntimeException(\"case_{index}\");",
                ]
            )
        lines.extend(["    }", "}"])
        return ("\n".join(lines),)
    if normalized == "javascript":
        lines = [
            "const _assertEqual = (actual, expected, label) => {",
            "  if (actual !== expected) {",
            "    throw new Error(label + ':' + JSON.stringify(actual) + '!=' + JSON.stringify(expected));",
            "  }",
            "};",
        ]
        for index, case in enumerate(cases):
            args = ", ".join(literal("javascript", arg) for arg in case["args"])
            expected = literal("javascript", case["expected"])
            lines.append(f"_assertEqual({entry_point}({args}), {expected}, 'case_{index}');")
        return ("\n".join(lines),)
    if normalized == "go":
        lines = ["package main", "import \"fmt\"", "func main() {"]
        for index, case in enumerate(cases):
            args = ", ".join(literal("go", arg) for arg in case["args"])
            expected = literal("go", case["expected"])
            lines.extend(
                [
                    f"    actual{index} := {entry_point}({args})",
                    f"    expected{index} := {expected}",
                    f"    if actual{index} != expected{index} {{ panic(fmt.Sprintf(\"case_{index}:%v!=%v\", actual{index}, expected{index})) }}",
                ]
            )
        lines.append("}")
        return ("\n".join(lines),)
    return ()


def execution_tests(language: str, entry_point: str, base_cases: list[dict[str, Any]], stress_cases: list[dict[str, Any]]) -> tuple[str, ...]:
    return reference_tests(language, entry_point, list(base_cases) + list(stress_cases))


def strings_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
def {name}(text):
    parts = []
    current = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return "/".join(parts)
""".strip(),
        "cpp": f"""
#include <cctype>
#include <string>
#include <vector>
using namespace std;
string {name}(const string& text) {{
    vector<string> parts;
    string current;
    for (char raw : text) {{
        unsigned char ch = static_cast<unsigned char>(raw);
        if (isalnum(ch)) current.push_back(static_cast<char>(tolower(ch)));
        else if (!current.empty()) {{
            parts.push_back(current);
            current.clear();
        }}
    }}
    if (!current.empty()) parts.push_back(current);
    string out;
    for (size_t index = 0; index < parts.size(); ++index) {{
        if (index > 0) out += "/";
        out += parts[index];
    }}
    return out;
}}
""".strip(),
        "java": f"""
class Solution {{
    public static String {name}(String text) {{
        StringBuilder current = new StringBuilder();
        java.util.List<String> parts = new java.util.ArrayList<>();
        for (int index = 0; index < text.length(); index++) {{
            char ch = Character.toLowerCase(text.charAt(index));
            if (Character.isLetterOrDigit(ch)) current.append(ch);
            else if (current.length() > 0) {{
                parts.add(current.toString());
                current.setLength(0);
            }}
        }}
        if (current.length() > 0) parts.add(current.toString());
        return String.join("/", parts);
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(text) {{
  const parts = [];
  let current = "";
  for (const raw of text.toLowerCase()) {{
    if (/[a-z0-9]/.test(raw)) current += raw;
    else if (current.length > 0) {{
      parts.push(current);
      current = "";
    }}
  }}
  if (current.length > 0) parts.push(current);
  return parts.join("/");
}}
""".strip(),
        "go": f"""
package main

import (
    "strings"
    "unicode"
)

func {name}(text string) string {{
    parts := make([]string, 0)
    current := make([]rune, 0)
    for _, raw := range strings.ToLower(text) {{
        if unicode.IsLetter(raw) || unicode.IsDigit(raw) {{
            current = append(current, raw)
        }} else if len(current) > 0 {{
            parts = append(parts, string(current))
            current = current[:0]
        }}
    }}
    if len(current) > 0 {{
        parts = append(parts, string(current))
    }}
    return strings.Join(parts, "/")
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def arrays_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
def {name}(values, marker):
    seen = False
    total = 0
    for value in values:
        if value == marker:
            seen = True
            continue
        if seen and value > 0:
            total += value
    return total
""".strip(),
        "cpp": f"""
#include <vector>
using namespace std;
int {name}(const vector<int>& values, int marker) {{
    bool seen = false;
    int total = 0;
    for (int value : values) {{
        if (value == marker) {{
            seen = true;
            continue;
        }}
        if (seen && value > 0) total += value;
    }}
    return total;
}}
""".strip(),
        "java": f"""
import java.util.*;
class Solution {{
    public static int {name}(List<Integer> values, int marker) {{
        boolean seen = false;
        int total = 0;
        for (int value : values) {{
            if (value == marker) {{
                seen = true;
                continue;
            }}
            if (seen && value > 0) total += value;
        }}
        return total;
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(values, marker) {{
  let seen = false;
  let total = 0;
  for (const value of values) {{
    if (value === marker) {{
      seen = true;
      continue;
    }}
    if (seen && value > 0) total += value;
  }}
  return total;
}}
""".strip(),
        "go": f"""
package main

func {name}(values []int, marker int) int {{
    seen := false
    total := 0
    for _, value := range values {{
        if value == marker {{
            seen = true
            continue
        }}
        if seen && value > 0 {{
            total += value
        }}
    }}
    return total
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def maps_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
def {name}(entries):
    totals = {{}}
    for entry in entries:
        if ":" not in entry:
            continue
        key, raw_value = entry.split(":", 1)
        key = key.strip().lower()
        if not key:
            continue
        try:
            value = int(raw_value.strip())
        except ValueError:
            continue
        totals[key] = totals.get(key, 0) + value
    if not totals:
        return ""
    best_key = ""
    best_total = None
    for key in sorted(totals):
        total = totals[key]
        if best_total is None or total > best_total:
            best_key = key
            best_total = total
    return best_key
""".strip(),
        "cpp": f"""
#include <cctype>
#include <map>
#include <string>
#include <vector>
using namespace std;
string {name}(const vector<string>& entries) {{
    map<string, int> totals;
    for (const string& entry : entries) {{
        size_t pos = entry.find(':');
        if (pos == string::npos) continue;
        string key = entry.substr(0, pos);
        string value_text = entry.substr(pos + 1);
        string normalized;
        for (char raw : key) {{
            unsigned char ch = static_cast<unsigned char>(raw);
            if (!isspace(ch)) normalized.push_back(static_cast<char>(tolower(ch)));
        }}
        if (normalized.empty()) continue;
        try {{
            int value = stoi(value_text);
            totals[normalized] += value;
        }} catch (...) {{
            continue;
        }}
    }}
    string best;
    bool initialized = false;
    int best_total = 0;
    for (const auto& item : totals) {{
        if (!initialized || item.second > best_total) {{
            initialized = true;
            best = item.first;
            best_total = item.second;
        }}
    }}
    return initialized ? best : "";
}}
""".strip(),
        "java": f"""
import java.util.*;
class Solution {{
    public static String {name}(List<String> entries) {{
        Map<String, Integer> totals = new TreeMap<>();
        for (String entry : entries) {{
            int pos = entry.indexOf(':');
            if (pos < 0) continue;
            String key = entry.substring(0, pos).trim().toLowerCase();
            if (key.isEmpty()) continue;
            try {{
                int value = Integer.parseInt(entry.substring(pos + 1).trim());
                totals.put(key, totals.getOrDefault(key, 0) + value);
            }} catch (NumberFormatException ignored) {{
            }}
        }}
        String bestKey = "";
        Integer bestTotal = null;
        for (Map.Entry<String, Integer> item : totals.entrySet()) {{
            if (bestTotal == null || item.getValue() > bestTotal) {{
                bestKey = item.getKey();
                bestTotal = item.getValue();
            }}
        }}
        return bestKey;
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(entries) {{
  const totals = new Map();
  for (const entry of entries) {{
    const pos = entry.indexOf(":");
    if (pos < 0) continue;
    const key = entry.slice(0, pos).trim().toLowerCase();
    if (!key) continue;
    const value = Number.parseInt(entry.slice(pos + 1).trim(), 10);
    if (Number.isNaN(value)) continue;
    totals.set(key, (totals.get(key) || 0) + value);
  }}
  let bestKey = "";
  let bestTotal = null;
  for (const key of Array.from(totals.keys()).sort()) {{
    const total = totals.get(key);
    if (bestTotal === null || total > bestTotal) {{
      bestKey = key;
      bestTotal = total;
    }}
  }}
  return bestKey;
}}
""".strip(),
        "go": f"""
package main

import (
    "sort"
    "strconv"
    "strings"
)

func {name}(entries []string) string {{
    totals := map[string]int{{}}
    for _, entry := range entries {{
        pos := strings.Index(entry, ":")
        if pos < 0 {{
            continue
        }}
        key := strings.ToLower(strings.TrimSpace(entry[:pos]))
        if key == "" {{
            continue
        }}
        value, err := strconv.Atoi(strings.TrimSpace(entry[pos+1:]))
        if err != nil {{
            continue
        }}
        totals[key] += value
    }}
    keys := make([]string, 0, len(totals))
    for key := range totals {{
        keys = append(keys, key)
    }}
    sort.Strings(keys)
    bestKey := ""
    initialized := false
    bestTotal := 0
    for _, key := range keys {{
        total := totals[key]
        if !initialized || total > bestTotal {{
            bestKey = key
            bestTotal = total
            initialized = true
        }}
    }}
    if initialized {{
        return bestKey
    }}
    return ""
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def parsing_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
def {name}(lines, window):
    count = 0
    for line in lines:
        if "," not in line:
            continue
        left, right = line.split(",", 1)
        try:
            first = int(left.strip())
            second = int(right.strip())
        except ValueError:
            continue
        if abs(first - second) <= window and first % 2 == second % 2:
            count += 1
    return count
""".strip(),
        "cpp": f"""
#include <cmath>
#include <string>
#include <vector>
using namespace std;
int {name}(const vector<string>& lines, int window) {{
    int count = 0;
    for (const string& line : lines) {{
        size_t pos = line.find(',');
        if (pos == string::npos) continue;
        try {{
            int left = stoi(line.substr(0, pos));
            int right = stoi(line.substr(pos + 1));
            if (abs(left - right) <= window && (left % 2) == (right % 2)) count += 1;
        }} catch (...) {{
            continue;
        }}
    }}
    return count;
}}
""".strip(),
        "java": f"""
import java.util.*;
class Solution {{
    public static int {name}(List<String> lines, int window) {{
        int count = 0;
        for (String line : lines) {{
            int pos = line.indexOf(',');
            if (pos < 0) continue;
            try {{
                int left = Integer.parseInt(line.substring(0, pos).trim());
                int right = Integer.parseInt(line.substring(pos + 1).trim());
                if (Math.abs(left - right) <= window && left % 2 == right % 2) count++;
            }} catch (NumberFormatException ignored) {{
            }}
        }}
        return count;
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(lines, window) {{
  let count = 0;
  for (const line of lines) {{
    const pos = line.indexOf(",");
    if (pos < 0) continue;
    const left = Number.parseInt(line.slice(0, pos).trim(), 10);
    const right = Number.parseInt(line.slice(pos + 1).trim(), 10);
    if (Number.isNaN(left) || Number.isNaN(right)) continue;
    if (Math.abs(left - right) <= window && left % 2 === right % 2) count += 1;
  }}
  return count;
}}
""".strip(),
        "go": f"""
package main

import (
    "math"
    "strconv"
    "strings"
)

func {name}(lines []string, window int) int {{
    count := 0
    for _, line := range lines {{
        pos := strings.Index(line, ",")
        if pos < 0 {{
            continue
        }}
        left, errLeft := strconv.Atoi(strings.TrimSpace(line[:pos]))
        right, errRight := strconv.Atoi(strings.TrimSpace(line[pos+1:]))
        if errLeft != nil || errRight != nil {{
            continue
        }}
        if int(math.Abs(float64(left-right))) <= window && left%2 == right%2 {{
            count++
        }}
    }}
    return count
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def bit_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
def {name}(values):
    score = 0
    for value in values:
        bits = abs(value).bit_count()
        if value % 2 == 0:
            score += bits
        else:
            score -= bits
    return score
""".strip(),
        "cpp": f"""
#include <cstdlib>
#include <vector>
using namespace std;
int {name}(const vector<int>& values) {{
    int score = 0;
    for (int value : values) {{
        int bits = __builtin_popcount(static_cast<unsigned int>(abs(value)));
        if (value % 2 == 0) score += bits;
        else score -= bits;
    }}
    return score;
}}
""".strip(),
        "java": f"""
import java.util.*;
class Solution {{
    public static int {name}(List<Integer> values) {{
        int score = 0;
        for (int value : values) {{
            int bits = Integer.bitCount(Math.abs(value));
            if (value % 2 == 0) score += bits;
            else score -= bits;
        }}
        return score;
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(values) {{
  let score = 0;
  for (const value of values) {{
    let current = Math.abs(value);
    let bits = 0;
    while (current > 0) {{
      bits += current & 1;
      current >>= 1;
    }}
    if (value % 2 === 0) score += bits;
    else score -= bits;
  }}
  return score;
}}
""".strip(),
        "go": f"""
package main

import "math/bits"

func {name}(values []int) int {{
    score := 0
    for _, value := range values {{
        current := value
        if current < 0 {{
            current = -current
        }}
        count := bits.OnesCount(uint(current))
        if value%2 == 0 {{
            score += count
        }} else {{
            score -= count
        }}
    }}
    return score
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def interval_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
def {name}(intervals):
    pairs = []
    for item in intervals:
        if "-" not in item:
            continue
        left, right = item.split("-", 1)
        try:
            start = int(left.strip())
            end = int(right.strip())
        except ValueError:
            continue
        if start > end:
            start, end = end, start
        pairs.append((start, end))
    if not pairs:
        return 0
    pairs.sort()
    total = 0
    current_start, current_end = pairs[0]
    for start, end in pairs[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start + 1
            current_start, current_end = start, end
    total += current_end - current_start + 1
    return total
""".strip(),
        "cpp": f"""
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
using namespace std;
int {name}(const vector<string>& intervals) {{
    vector<pair<int, int>> pairs;
    for (const string& item : intervals) {{
        size_t pos = item.find('-');
        if (pos == string::npos) continue;
        try {{
            int start = stoi(item.substr(0, pos));
            int end = stoi(item.substr(pos + 1));
            if (start > end) swap(start, end);
            pairs.push_back({{start, end}});
        }} catch (...) {{
            continue;
        }}
    }}
    if (pairs.empty()) return 0;
    sort(pairs.begin(), pairs.end());
    int total = 0;
    int current_start = pairs[0].first;
    int current_end = pairs[0].second;
    for (size_t index = 1; index < pairs.size(); ++index) {{
        int start = pairs[index].first;
        int end = pairs[index].second;
        if (start <= current_end + 1) current_end = max(current_end, end);
        else {{
            total += current_end - current_start + 1;
            current_start = start;
            current_end = end;
        }}
    }}
    total += current_end - current_start + 1;
    return total;
}}
""".strip(),
        "java": f"""
import java.util.*;
class Solution {{
    public static int {name}(List<String> intervals) {{
        List<int[]> pairs = new ArrayList<>();
        for (String item : intervals) {{
            int pos = item.indexOf('-');
            if (pos < 0) continue;
            try {{
                int start = Integer.parseInt(item.substring(0, pos).trim());
                int end = Integer.parseInt(item.substring(pos + 1).trim());
                if (start > end) {{
                    int temp = start;
                    start = end;
                    end = temp;
                }}
                pairs.add(new int[]{{start, end}});
            }} catch (NumberFormatException ignored) {{
            }}
        }}
        if (pairs.isEmpty()) return 0;
        pairs.sort(Comparator.comparingInt(item -> item[0]));
        int total = 0;
        int currentStart = pairs.get(0)[0];
        int currentEnd = pairs.get(0)[1];
        for (int index = 1; index < pairs.size(); index++) {{
            int[] pair = pairs.get(index);
            if (pair[0] <= currentEnd + 1) currentEnd = Math.max(currentEnd, pair[1]);
            else {{
                total += currentEnd - currentStart + 1;
                currentStart = pair[0];
                currentEnd = pair[1];
            }}
        }}
        total += currentEnd - currentStart + 1;
        return total;
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(intervals) {{
  const pairs = [];
  for (const item of intervals) {{
    const pos = item.indexOf("-");
    if (pos < 0) continue;
    const startRaw = Number.parseInt(item.slice(0, pos).trim(), 10);
    const endRaw = Number.parseInt(item.slice(pos + 1).trim(), 10);
    if (Number.isNaN(startRaw) || Number.isNaN(endRaw)) continue;
    pairs.push([Math.min(startRaw, endRaw), Math.max(startRaw, endRaw)]);
  }}
  if (pairs.length === 0) return 0;
  pairs.sort((left, right) => left[0] - right[0]);
  let total = 0;
  let [currentStart, currentEnd] = pairs[0];
  for (const [start, end] of pairs.slice(1)) {{
    if (start <= currentEnd + 1) currentEnd = Math.max(currentEnd, end);
    else {{
      total += currentEnd - currentStart + 1;
      currentStart = start;
      currentEnd = end;
    }}
  }}
  total += currentEnd - currentStart + 1;
  return total;
}}
""".strip(),
        "go": f"""
package main

import (
    "sort"
    "strconv"
    "strings"
)

func {name}(intervals []string) int {{
    pairs := make([][2]int, 0)
    for _, item := range intervals {{
        pos := strings.Index(item, "-")
        if pos < 0 {{
            continue
        }}
        start, errStart := strconv.Atoi(strings.TrimSpace(item[:pos]))
        end, errEnd := strconv.Atoi(strings.TrimSpace(item[pos+1:]))
        if errStart != nil || errEnd != nil {{
            continue
        }}
        if start > end {{
            start, end = end, start
        }}
        pairs = append(pairs, [2]int{{start, end}})
    }}
    if len(pairs) == 0 {{
        return 0
    }}
    sort.Slice(pairs, func(i, j int) bool {{
        return pairs[i][0] < pairs[j][0]
    }})
    total := 0
    currentStart := pairs[0][0]
    currentEnd := pairs[0][1]
    for _, pair := range pairs[1:] {{
        if pair[0] <= currentEnd+1 {{
            if pair[1] > currentEnd {{
                currentEnd = pair[1]
            }}
        }} else {{
            total += currentEnd - currentStart + 1
            currentStart = pair[0]
            currentEnd = pair[1]
        }}
    }}
    total += currentEnd - currentStart + 1
    return total
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def graph_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
from collections import deque

def {name}(grid):
    start = None
    end = None
    for row, line in enumerate(grid):
        for col, ch in enumerate(line):
            if ch == "S":
                start = (row, col)
            elif ch == "E":
                end = (row, col)
    if start is None or end is None:
        return -1
    queue = deque([(start[0], start[1], 0)])
    seen = {{start}}
    while queue:
        row, col, dist = queue.popleft()
        if (row, col) == end:
            return dist
        for row_delta, col_delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row + row_delta
            next_col = col + col_delta
            if 0 <= next_row < len(grid) and 0 <= next_col < len(grid[next_row]):
                if grid[next_row][next_col] != "#" and (next_row, next_col) not in seen:
                    seen.add((next_row, next_col))
                    queue.append((next_row, next_col, dist + 1))
    return -1
""".strip(),
        "cpp": f"""
#include <queue>
#include <tuple>
#include <vector>
#include <string>
using namespace std;
int {name}(const vector<string>& grid) {{
    pair<int, int> start = {{-1, -1}};
    pair<int, int> goal = {{-1, -1}};
    for (int row = 0; row < static_cast<int>(grid.size()); ++row) {{
        for (int col = 0; col < static_cast<int>(grid[row].size()); ++col) {{
            if (grid[row][col] == 'S') start = {{row, col}};
            if (grid[row][col] == 'E') goal = {{row, col}};
        }}
    }}
    if (start.first < 0 || goal.first < 0) return -1;
    queue<tuple<int, int, int>> todo;
    vector<vector<int>> seen(grid.size());
    for (int row = 0; row < static_cast<int>(grid.size()); ++row) seen[row].assign(grid[row].size(), 0);
    todo.push({{start.first, start.second, 0}});
    seen[start.first][start.second] = 1;
    int deltas[4][2] = {{ {{1, 0}}, {{-1, 0}}, {{0, 1}}, {{0, -1}} }};
    while (!todo.empty()) {{
        auto [row, col, dist] = todo.front();
        todo.pop();
        if (row == goal.first && col == goal.second) return dist;
        for (auto& delta : deltas) {{
            int next_row = row + delta[0];
            int next_col = col + delta[1];
            if (0 <= next_row && next_row < static_cast<int>(grid.size()) && 0 <= next_col && next_col < static_cast<int>(grid[next_row].size())) {{
                if (grid[next_row][next_col] != '#' && !seen[next_row][next_col]) {{
                    seen[next_row][next_col] = 1;
                    todo.push({{next_row, next_col, dist + 1}});
                }}
            }}
        }}
    }}
    return -1;
}}
""".strip(),
        "java": f"""
import java.util.*;
class Solution {{
    public static int {name}(List<String> grid) {{
        int startRow = -1;
        int startCol = -1;
        int endRow = -1;
        int endCol = -1;
        for (int row = 0; row < grid.size(); row++) {{
            String line = grid.get(row);
            for (int col = 0; col < line.length(); col++) {{
                if (line.charAt(col) == 'S') {{
                    startRow = row;
                    startCol = col;
                }} else if (line.charAt(col) == 'E') {{
                    endRow = row;
                    endCol = col;
                }}
            }}
        }}
        if (startRow < 0 || endRow < 0) return -1;
        boolean[][] seen = new boolean[grid.size()][];
        for (int row = 0; row < grid.size(); row++) seen[row] = new boolean[grid.get(row).length()];
        Queue<int[]> queue = new ArrayDeque<>();
        queue.add(new int[]{{startRow, startCol, 0}});
        seen[startRow][startCol] = true;
        int[][] deltas = new int[][]{{{{1, 0}}, {{-1, 0}}, {{0, 1}}, {{0, -1}}}};
        while (!queue.isEmpty()) {{
            int[] current = queue.remove();
            if (current[0] == endRow && current[1] == endCol) return current[2];
            for (int[] delta : deltas) {{
                int nextRow = current[0] + delta[0];
                int nextCol = current[1] + delta[1];
                if (0 <= nextRow && nextRow < grid.size() && 0 <= nextCol && nextCol < grid.get(nextRow).length()) {{
                    if (grid.get(nextRow).charAt(nextCol) != '#' && !seen[nextRow][nextCol]) {{
                        seen[nextRow][nextCol] = true;
                        queue.add(new int[]{{nextRow, nextCol, current[2] + 1}});
                    }}
                }}
            }}
        }}
        return -1;
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(grid) {{
  let start = null;
  let end = null;
  for (let row = 0; row < grid.length; row += 1) {{
    for (let col = 0; col < grid[row].length; col += 1) {{
      if (grid[row][col] === "S") start = [row, col];
      if (grid[row][col] === "E") end = [row, col];
    }}
  }}
  if (!start || !end) return -1;
  const queue = [[start[0], start[1], 0]];
  const seen = new Set([`${{start[0]}}:${{start[1]}}`]);
  const deltas = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  while (queue.length > 0) {{
    const [row, col, dist] = queue.shift();
    if (row === end[0] && col === end[1]) return dist;
    for (const [rowDelta, colDelta] of deltas) {{
      const nextRow = row + rowDelta;
      const nextCol = col + colDelta;
      if (0 <= nextRow && nextRow < grid.length && 0 <= nextCol && nextCol < grid[nextRow].length) {{
        if (grid[nextRow][nextCol] !== "#" && !seen.has(`${{nextRow}}:${{nextCol}}`)) {{
          seen.add(`${{nextRow}}:${{nextCol}}`);
          queue.push([nextRow, nextCol, dist + 1]);
        }}
      }}
    }}
  }}
  return -1;
}}
""".strip(),
        "go": f"""
package main

type queueItem struct {{
    row int
    col int
    dist int
}}

func {name}(grid []string) int {{
    startRow, startCol := -1, -1
    endRow, endCol := -1, -1
    for row, line := range grid {{
        for col, ch := range line {{
            if ch == 'S' {{
                startRow, startCol = row, col
            }} else if ch == 'E' {{
                endRow, endCol = row, col
            }}
        }}
    }}
    if startRow < 0 || endRow < 0 {{
        return -1
    }}
    seen := map[[2]int]bool{{{{startRow, startCol}}: true}}
    queue := []queueItem{{{{startRow, startCol, 0}}}}
    deltas := [][2]int{{{{1, 0}}, {{-1, 0}}, {{0, 1}}, {{0, -1}}}}
    for len(queue) > 0 {{
        current := queue[0]
        queue = queue[1:]
        if current.row == endRow && current.col == endCol {{
            return current.dist
        }}
        for _, delta := range deltas {{
            nextRow := current.row + delta[0]
            nextCol := current.col + delta[1]
            if nextRow >= 0 && nextRow < len(grid) && nextCol >= 0 && nextCol < len(grid[nextRow]) {{
                if grid[nextRow][nextCol] != '#' {{
                    key := [2]int{{nextRow, nextCol}}
                    if !seen[key] {{
                        seen[key] = true
                        queue = append(queue, queueItem{{nextRow, nextCol, current.dist + 1}})
                    }}
                }}
            }}
        }}
    }}
    return -1
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def dp_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
def {name}(values):
    include = 0
    exclude = 0
    for value in values:
        include, exclude = exclude + value, max(include, exclude)
    return max(include, exclude, 0)
""".strip(),
        "cpp": f"""
#include <algorithm>
#include <vector>
using namespace std;
int {name}(const vector<int>& values) {{
    int include = 0;
    int exclude = 0;
    for (int value : values) {{
        int next_include = exclude + value;
        int next_exclude = max(include, exclude);
        include = next_include;
        exclude = next_exclude;
    }}
    return max(max(include, exclude), 0);
}}
""".strip(),
        "java": f"""
import java.util.*;
class Solution {{
    public static int {name}(List<Integer> values) {{
        int include = 0;
        int exclude = 0;
        for (int value : values) {{
            int nextInclude = exclude + value;
            int nextExclude = Math.max(include, exclude);
            include = nextInclude;
            exclude = nextExclude;
        }}
        return Math.max(Math.max(include, exclude), 0);
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(values) {{
  let include = 0;
  let exclude = 0;
  for (const value of values) {{
    const nextInclude = exclude + value;
    const nextExclude = Math.max(include, exclude);
    include = nextInclude;
    exclude = nextExclude;
  }}
  return Math.max(include, exclude, 0);
}}
""".strip(),
        "go": f"""
package main

func {name}(values []int) int {{
    include := 0
    exclude := 0
    for _, value := range values {{
        nextInclude := exclude + value
        nextExclude := exclude
        if include > nextExclude {{
            nextExclude = include
        }}
        include = nextInclude
        exclude = nextExclude
    }}
    result := include
    if exclude > result {{
        result = exclude
    }}
    if result < 0 {{
        return 0
    }}
    return result
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def stateful_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
def {name}(events):
    inventory = {{}}
    for event in events:
        parts = event.split()
        if len(parts) != 3:
            continue
        action, item, raw_count = parts
        try:
            count = int(raw_count)
        except ValueError:
            continue
        if count < 0:
            continue
        if action == "add":
            inventory[item] = inventory.get(item, 0) + count
        elif action == "remove":
            inventory[item] = max(0, inventory.get(item, 0) - count)
    return sum(inventory.values())
""".strip(),
        "cpp": f"""
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
int {name}(const vector<string>& events) {{
    map<string, int> inventory;
    for (const string& event : events) {{
        stringstream stream(event);
        string action;
        string item;
        int count = 0;
        if (!(stream >> action >> item >> count)) continue;
        if (count < 0) continue;
        if (action == "add") inventory[item] += count;
        else if (action == "remove") inventory[item] = max(0, inventory[item] - count);
    }}
    int total = 0;
    for (const auto& item : inventory) total += item.second;
    return total;
}}
""".strip(),
        "java": f"""
import java.util.*;
class Solution {{
    public static int {name}(List<String> events) {{
        Map<String, Integer> inventory = new HashMap<>();
        for (String event : events) {{
            String[] parts = event.split("\\\\s+");
            if (parts.length != 3) continue;
            try {{
                int count = Integer.parseInt(parts[2]);
                if (count < 0) continue;
                if (parts[0].equals("add")) inventory.put(parts[1], inventory.getOrDefault(parts[1], 0) + count);
                else if (parts[0].equals("remove")) inventory.put(parts[1], Math.max(0, inventory.getOrDefault(parts[1], 0) - count));
            }} catch (NumberFormatException ignored) {{
            }}
        }}
        int total = 0;
        for (int value : inventory.values()) total += value;
        return total;
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(events) {{
  const inventory = new Map();
  for (const event of events) {{
    const parts = event.trim().split(/\\s+/);
    if (parts.length !== 3) continue;
    const count = Number.parseInt(parts[2], 10);
    if (Number.isNaN(count) || count < 0) continue;
    const current = inventory.get(parts[1]) || 0;
    if (parts[0] === "add") inventory.set(parts[1], current + count);
    else if (parts[0] === "remove") inventory.set(parts[1], Math.max(0, current - count));
  }}
  let total = 0;
  for (const value of inventory.values()) total += value;
  return total;
}}
""".strip(),
        "go": f"""
package main

import (
    "strconv"
    "strings"
)

func {name}(events []string) int {{
    inventory := map[string]int{{}}
    for _, event := range events {{
        parts := strings.Fields(event)
        if len(parts) != 3 {{
            continue
        }}
        count, err := strconv.Atoi(parts[2])
        if err != nil || count < 0 {{
            continue
        }}
        current := inventory[parts[1]]
        if parts[0] == "add" {{
            inventory[parts[1]] = current + count
        }} else if parts[0] == "remove" {{
            if current-count < 0 {{
                inventory[parts[1]] = 0
            }} else {{
                inventory[parts[1]] = current - count
            }}
        }}
    }}
    total := 0
    for _, value := range inventory {{
        total += value
    }}
    return total
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def api_source(language: str, name: str) -> str:
    templates = {
        "python": f"""
def {name}(query):
    latest = {{}}
    for part in query.replace(";", "&").split("&"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key and value:
            latest[key] = value
    return "&".join(f"{{key}}={{latest[key]}}" for key in sorted(latest))
""".strip(),
        "cpp": f"""
#include <algorithm>
#include <map>
#include <string>
using namespace std;
string {name}(string query) {{
    replace(query.begin(), query.end(), ';', '&');
    map<string, string> latest;
    size_t start = 0;
    while (start <= query.size()) {{
        size_t end = query.find('&', start);
        string part = query.substr(start, end == string::npos ? string::npos : end - start);
        size_t pos = part.find('=');
        if (pos != string::npos) {{
            string key = part.substr(0, pos);
            string value = part.substr(pos + 1);
            transform(key.begin(), key.end(), key.begin(), ::tolower);
            transform(value.begin(), value.end(), value.begin(), ::tolower);
            if (!key.empty() && !value.empty()) latest[key] = value;
        }}
        if (end == string::npos) break;
        start = end + 1;
    }}
    string out;
    for (const auto& item : latest) {{
        if (!out.empty()) out += "&";
        out += item.first + "=" + item.second;
    }}
    return out;
}}
""".strip(),
        "java": f"""
import java.util.*;
class Solution {{
    public static String {name}(String query) {{
        Map<String, String> latest = new TreeMap<>();
        for (String part : query.replace(';', '&').split("&")) {{
            int pos = part.indexOf('=');
            if (pos < 0) continue;
            String key = part.substring(0, pos).trim().toLowerCase();
            String value = part.substring(pos + 1).trim().toLowerCase();
            if (!key.isEmpty() && !value.isEmpty()) latest.put(key, value);
        }}
        List<String> pairs = new ArrayList<>();
        for (Map.Entry<String, String> item : latest.entrySet()) pairs.add(item.getKey() + "=" + item.getValue());
        return String.join("&", pairs);
    }}
}}
""".strip(),
        "javascript": f"""
function {name}(query) {{
  const latest = new Map();
  for (const part of query.replace(/;/g, "&").split("&")) {{
    const pos = part.indexOf("=");
    if (pos < 0) continue;
    const key = part.slice(0, pos).trim().toLowerCase();
    const value = part.slice(pos + 1).trim().toLowerCase();
    if (key && value) latest.set(key, value);
  }}
  return Array.from(latest.keys()).sort().map((key) => `${{key}}=${{latest.get(key)}}`).join("&");
}}
""".strip(),
        "go": f"""
package main

import (
    "sort"
    "strings"
)

func {name}(query string) string {{
    latest := map[string]string{{}}
    normalized := strings.ReplaceAll(query, ";", "&")
    for _, part := range strings.Split(normalized, "&") {{
        pos := strings.Index(part, "=")
        if pos < 0 {{
            continue
        }}
        key := strings.ToLower(strings.TrimSpace(part[:pos]))
        value := strings.ToLower(strings.TrimSpace(part[pos+1:]))
        if key != "" && value != "" {{
            latest[key] = value
        }}
    }}
    keys := make([]string, 0, len(latest))
    for key := range latest {{
        keys = append(keys, key)
    }}
    sort.Strings(keys)
    pairs := make([]string, 0, len(keys))
    for _, key := range keys {{
        pairs = append(pairs, key+"="+latest[key])
    }}
    return strings.Join(pairs, "&")
}}
""".strip(),
    }
    return templates[normalize_language_name(language)]


def solution_source(category: str, language: str, name: str) -> str:
    builders = {
        "strings": strings_source,
        "arrays/lists": arrays_source,
        "maps/sets": maps_source,
        "parsing": parsing_source,
        "math/bit ops": bit_source,
        "interval/greedy": interval_source,
        "graph/search": graph_source,
        "dp/recursion": dp_source,
        "stateful update": stateful_source,
        "API-style normalization": api_source,
    }
    return builders[category](language, name)
