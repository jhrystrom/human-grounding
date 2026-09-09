import pytest
from conftest import drag, grab_points, make_csv


@pytest.mark.parametrize("size", [21, 30, 50])
def test_loads_more_than_twenty_statements(canvas, upload_csv, size):
    """size > 20 must render exactly `size` cards, not the default 20."""
    upload_csv(canvas, make_csv(100))
    assert canvas.locator("#status").inner_text() == "100 statements loaded"

    canvas.fill("#size", str(size))
    canvas.click("#load")

    assert canvas.locator(".stmt").count() == size
    assert canvas.locator("#status").inner_text() == f"0 / {size} placed on canvas"


def test_size_larger_than_dataset_is_clamped(canvas, upload_csv):
    upload_csv(canvas, make_csv(25))

    canvas.fill("#size", "40")
    canvas.click("#load")

    assert canvas.locator(".stmt").count() == 25


def test_loaded_statements_have_ids_and_text(canvas, upload_csv):
    upload_csv(canvas, make_csv(100))
    canvas.fill("#size", "30")
    canvas.click("#load")

    cards = canvas.locator(".stmt")
    assert cards.count() == 30
    for i in range(cards.count()):
        text = cards.nth(i).inner_text()
        assert text.startswith("#")
        assert "undefined" not in text
        assert "Statement number" in text


@pytest.mark.parametrize(
    ("columns", "expected_missing"),
    [
        (("statement_id", "dataset", "body"), '"text"'),
        (("id", "dataset", "text"), '"statement_id"'),
        (("foo", "bar"), '"statement_id" and "text"'),
    ],
)
def test_missing_column_shows_useful_error(
    canvas, upload_csv, columns, expected_missing
):
    upload_csv(canvas, make_csv(5, columns=columns))

    error = canvas.locator("#error")
    assert error.is_visible()
    assert expected_missing in error.inner_text()
    # the columns that *were* found are named, so the user can spot the typo
    for column in columns:
        assert f'"{column}"' in error.inner_text()


def test_missing_column_blocks_loading(canvas, upload_csv):
    upload_csv(canvas, make_csv(5, columns=("statement_id", "dataset", "body")))

    canvas.on("dialog", lambda dialog: dialog.accept())
    canvas.click("#load")

    assert canvas.locator(".stmt").count() == 0
    assert canvas.locator("#save").is_disabled()


def test_valid_csv_shows_no_error(canvas, upload_csv):
    upload_csv(canvas, make_csv(5))

    assert canvas.locator("#error").inner_text() == ""
    assert canvas.locator("#status").inner_text() == "5 statements loaded"


def test_extra_columns_are_fine(canvas, upload_csv):
    upload_csv(
        canvas, make_csv(5, columns=("statement_id", "dataset", "text", "Agency"))
    )

    assert canvas.locator("#error").inner_text() == ""
    assert canvas.locator("#status").inner_text() == "5 statements loaded"


# --- Overlapping tray layout -------------------------------------------------


@pytest.mark.parametrize("size", [21, 40, 80])
def test_overlapped_cards_start_unplaced(canvas, upload_csv, size):
    """Overlapping may overhang the divider, but nothing may count as placed yet."""
    upload_csv(canvas, make_csv(100, text_len=400))
    canvas.fill("#size", str(size))
    canvas.click("#load")

    assert canvas.locator("#status").inner_text() == f"0 / {size} placed on canvas"
    assert canvas.locator(".stmt.placed").count() == 0


@pytest.mark.parametrize("size", [21, 40, 80])
def test_mixed_height_cards_start_unplaced_and_reachable(canvas, upload_csv, size):
    """Short cards have the highest centres, so they decide where the stack may start."""
    upload_csv(canvas, make_csv(100, vary=True))
    canvas.fill("#size", str(size))
    canvas.click("#load")

    assert canvas.locator("#status").inner_text() == f"0 / {size} placed on canvas"
    buried = [i for i, p in enumerate(grab_points(canvas)) if p is None]
    assert buried == [], f"cards with no clickable pixel: {buried}"


@pytest.mark.parametrize("size", [21, 40, 80])
def test_overlapping_cards_stay_inside_the_tray(canvas, upload_csv, size):
    """Above ~20 the grid no longer fits, so cards overlap rather than spill out."""
    upload_csv(canvas, make_csv(100))
    canvas.fill("#size", str(size))
    canvas.click("#load")

    box = canvas.evaluate(
        """() => {
          const m = document.querySelector('main').getBoundingClientRect();
          return [...document.querySelectorAll('.stmt')].map(e => {
            const r = e.getBoundingClientRect();
            return {top: r.top - m.top, bottom: r.bottom - m.top,
                    left: r.left - m.left, right: r.right - m.left};
          });
        }"""
    )
    main_h, main_w = canvas.evaluate(
        "() => [document.querySelector('main').clientHeight,"
        " document.querySelector('main').clientWidth]"
    )
    assert len(box) == size
    for r in box:
        assert r["top"] >= 0
        assert r["bottom"] <= main_h + 1
        assert r["left"] >= 0
        assert r["right"] <= main_w + 1


@pytest.mark.parametrize("size", [21, 40, 80])
@pytest.mark.parametrize("text_len", [20, 400])
def test_every_overlapped_card_is_reachable(canvas, upload_csv, size, text_len):
    """Overlap must be partial: no card may be completely buried."""
    upload_csv(canvas, make_csv(100, text_len=text_len))
    canvas.fill("#size", str(size))
    canvas.click("#load")

    points = grab_points(canvas)
    buried = [i for i, p in enumerate(points) if p is None]
    assert buried == [], f"cards with no clickable pixel: {buried}"


def test_cards_actually_overlap_above_twenty(canvas, upload_csv):
    upload_csv(canvas, make_csv(100))
    canvas.fill("#size", "40")
    canvas.click("#load")

    overlaps = canvas.evaluate(
        """() => {
          const rs = [...document.querySelectorAll('.stmt')].map(e => e.getBoundingClientRect());
          let n = 0;
          for (let i = 0; i < rs.length; i++)
            for (let j = i + 1; j < rs.length; j++)
              if (rs[i].right > rs[j].left && rs[j].right > rs[i].left &&
                  rs[i].bottom > rs[j].top && rs[j].bottom > rs[i].top) n++;
          return n;
        }"""
    )
    assert overlaps > 0


@pytest.mark.parametrize("index", [0, 19, 20, 39])
def test_overlapped_cards_can_be_dragged_onto_the_canvas(canvas, upload_csv, index):
    upload_csv(canvas, make_csv(100))
    canvas.fill("#size", "40")
    canvas.click("#load")

    card = canvas.locator(".stmt").nth(index)
    before = card.bounding_box()
    point = grab_points(canvas)[index]
    assert point is not None

    target = canvas.locator("#canvas").bounding_box()
    drag(
        canvas,
        point,
        target["x"] + target["width"] / 2,
        target["y"] + target["height"] / 2,
    )

    after = card.bounding_box()
    assert (after["x"], after["y"]) != (before["x"], before["y"])
    assert "placed" in (card.get_attribute("class") or "")
    assert canvas.locator("#status").inner_text() == "1 / 40 placed on canvas"


def test_all_forty_cards_can_be_dragged_one_by_one(canvas, upload_csv):
    """Drag every card out of the tray; each must be grabbable in turn."""
    upload_csv(canvas, make_csv(100))
    canvas.fill("#size", "40")
    canvas.click("#load")

    target = canvas.locator("#canvas").bounding_box()
    for i in range(40):
        point = grab_points(canvas)[i]
        assert point is not None, f"card {i} became unreachable"
        drag(
            canvas,
            point,
            target["x"] + 30 + (i % 10) * 60,
            target["y"] + 30 + (i // 10) * 60,
        )

    assert canvas.locator("#status").inner_text() == "40 / 40 placed on canvas"


def test_no_overlap_when_the_grid_fits(canvas, upload_csv):
    """Overlap is a fallback: 20 short cards still get a clean, spaced grid."""
    upload_csv(canvas, make_csv(100))
    canvas.fill("#size", "20")
    canvas.click("#load")

    overlaps = canvas.evaluate(
        """() => {
          const rs = [...document.querySelectorAll('.stmt')].map(e => e.getBoundingClientRect());
          let n = 0;
          for (let i = 0; i < rs.length; i++)
            for (let j = i + 1; j < rs.length; j++)
              if (rs[i].right > rs[j].left && rs[j].right > rs[i].left &&
                  rs[i].bottom > rs[j].top && rs[j].bottom > rs[i].top) n++;
          return n;
        }"""
    )
    assert overlaps == 0
