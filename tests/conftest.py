from pathlib import Path

import pytest

CANVAS_HTML = Path(__file__).resolve().parents[1] / "canvas.html"


@pytest.fixture
def canvas(page):
    """Open canvas.html from disk and return the Playwright page."""
    page.goto(CANVAS_HTML.as_uri())
    return page


@pytest.fixture
def upload_csv(tmp_path):
    """Write `content` to a temp CSV and feed it to the file input."""

    def _upload(page, content: str, name: str = "input.csv"):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        page.set_input_files("#file", str(path))
        return path

    return _upload


def make_csv(
    n: int,
    *,
    columns: tuple[str, ...] = ("statement_id", "dataset", "text"),
    text_len: int = 20,
    vary: bool = False,
) -> str:
    header = ",".join(columns)
    rows = []
    for i in range(n):
        # `vary` mimics real data, where card heights differ from row to row.
        length = (20 if i % 2 else 400) if vary else text_len
        text = f"Statement number {i} ".ljust(length, "x")
        values = {"statement_id": str(i), "dataset": "gov-ai", "text": text}
        rows.append(",".join(values.get(c, f"col-{c}-{i}") for c in columns))
    return "\n".join([header, *rows]) + "\n"


# Returns, for each .stmt card in DOM order, a viewport point where that card is the
# topmost element -- or null if it is completely buried and thus impossible to grab.
# Pass an index to scan just that one card, which keeps repeated grabs cheap.
GRAB_POINTS_JS = """
(only) => {
  const step = 3;
  const all = [...document.querySelectorAll('.stmt')];
  const cards = only === null ? all : [all[only]];
  return cards.map(card => {
    const r = card.getBoundingClientRect();
    for (let y = r.top + 1; y < r.bottom - 1; y += step) {
      for (let x = r.left + 1; x < r.right - 1; x += step) {
        const hit = document.elementFromPoint(x, y);
        if (hit && hit.closest('.stmt') === card) return { x, y };
      }
    }
    return null;
  });
}
"""


def grab_points(page):
    """A grabbable point for every card, in DOM order."""
    return page.evaluate(GRAB_POINTS_JS, None)


def grab_point(page, index):
    """A grabbable point for a single card."""
    return page.evaluate(GRAB_POINTS_JS, index)[0]


def drag(page, point, to_x, to_y):
    """One pointermove is all the drag handler needs, so keep the trip short."""
    page.mouse.move(point["x"], point["y"])
    page.mouse.down()
    page.mouse.move(to_x, to_y)
    page.mouse.up()
