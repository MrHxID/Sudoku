import io
import logging
import sys
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Iterator, Literal, overload


class bcolors:
    # Solution from https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def _assert_index(ind: int, group: Literal["row", "column", "block", "value"]) -> None:
    """Assert that the index is between ``0`` and ``8``.

    Args:
        ind (int): Index to validate.
        group (Literal[&quot;row&quot;, &quot;column&quot;, &quot;block&quot;, &quot;value&quot;]): Used for logging.

    Raises:
        AssertionError: If the index is invalid.
    """

    if group == "value":
        assert 1 <= ind <= 9, f"Invalid {group}; expected 1 <= {group} <= 9, got {ind}"
        return

    assert 0 <= ind <= 8, f"Invalid {group}; expected 0 <= {group} <= 8, got {ind}"


@dataclass
class Cell:
    column: int
    row: int
    block: int = field(repr=False)
    block_index: int = field(repr=False)
    candidates: list[int] = field(repr=False)
    collapsed: bool = False
    _value: int = field(default=0)

    def __init__(self, column: int, row: int) -> None:
        _assert_index(column, "column")
        _assert_index(row, "row")

        self.candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.column = column
        self.row = row

    @property
    def value(self) -> int:
        if not self._value:
            raise ValueError("This cell does not have a value yet.")

        return self._value

    @value.setter
    def value(self, val: int) -> None:
        if self._value:
            raise ValueError("This cell already has a value.")

        self._value = val
        self.collapsed = True
        self.candidates.clear()

    def format(self, spacing: int) -> tuple[str, str, str]:
        """Return a tuple of three formatted strings. Used in ``Sudoku.__str__``.

        Args:
            spacing (int): Width between each entry in the cell.

        Returns:
            tuple[str, str, str]: Format.
        """

        if self.collapsed:
            width = 4 * spacing + 1
            half_width = width >> 1
            return (
                width * " ",
                half_width * " "
                + bcolors.OKGREEN
                + str(self.value)
                + bcolors.ENDC
                + half_width * " ",
                width * " ",
            )

        output: list[str] = []

        for i in range(3):
            temp_str = (spacing >> 1) * " "

            for j in range(1, 4):
                temp_str += str(cand) if (cand := 3 * i + j) in self.candidates else " "
                temp_str += spacing * " "

            output.append(temp_str[:-1])

        return (output[0], output[1], output[2])

    def remove_candidate(self, candidate: int) -> bool:
        """Removes the ``candidate`` from this cells internal list. Does
        nothing if the candidate was already removed.

        Args:
            candidate (int): Candidate to remove.

        Returns:
            bool: Whether the candidate was removed.
        """

        if candidate in self.candidates:
            self.candidates.remove(candidate)
            return True

        return False


@dataclass
class Group:
    cells: tuple[Cell, ...]

    def __init__(self, cells: list[Cell] | tuple[Cell, ...]) -> None:
        self.cells = tuple(cells)

    def filter(self, *candidates: int) -> list[Cell]:
        """Returns a list of all the cells in this group
        that contain all of the ``candidates``.
        """

        cells = list(self.cells)

        for cand in candidates:
            cells = list(filter(lambda c: cand in c.candidates, cells))

        return cells

    def __iter__(self) -> Iterator[Cell]:
        return iter(self.cells)


class Row(Group): ...


class Column(Group): ...


class Block(Group): ...


class Sudoku:
    def __init__(self) -> None:
        cells: list[list[Cell]] = []
        flat_cells: list[Cell] = [0] * 81  # type: ignore

        columns: list[Column] = []
        rows: list[Row] = []
        blocks: list[Block] = []

        # Generate cells
        for row in range(9):
            cells.append([])
            for column in range(9):
                c = Cell(column=column, row=row)
                cells[row].append(c)
                flat_cells[9 * row + column] = c

            # Prepare cells and rows
            cells[row] = tuple(cells[row])  # type: ignore
            rows.append(Row(cells[row]))

        cells = tuple(cells)  # type: ignore

        # Prepare columns
        for column in range(9):
            temp_col: list[Cell] = []
            for row in range(9):
                temp_col.append(cells[row][column])

            columns.append(Column(temp_col))

        # Prepare blocks
        for block in range(9):
            temp_block: list[Cell] = []
            for row in range(9):
                for column in range(9):
                    if 3 * (row // 3) + column // 3 == block:
                        cells[row][column].block = block
                        cells[row][column].block_index = len(temp_block)
                        temp_block.append(cells[row][column])

            blocks.append(Block(temp_block))

        self.cells: tuple[tuple[Cell, ...], ...] = tuple(cells)  # type: ignore
        """2D-tuple containing all the cells of the sudoku. Indexing works left-to-right, top-to-bottom.
        """

        self.flat_cells: tuple[Cell, ...] = tuple(flat_cells)
        """1D-tuple containing all the cells of the sudoku. Indexing works left-to-right, top-to-bottom.
        """

        self.rows: tuple[Row, ...] = tuple(rows)
        """Tuple containing all the rows of the sudoku. Indexing works left-to-right.
        """

        self.columns: tuple[Column, ...] = tuple(columns)
        """Tuple containing all the columns of the sudoku. Indexing works top-to-bottom.
        """

        self.blocks: tuple[Block, ...] = tuple(blocks)
        """Tuple containing all the cells of the sudoku. Indexing works left-to-right, top-to-bottom.
        """

    def cell(self, column: int, row: int) -> Cell:
        """Return the cell in the given ``row`` and ``column``.

        Args:
            column (int): Column of the cell. Must be between ``0`` and ``8``
            row (int): Row of the cell. Must be between ``0`` and ``8``

        Returns:
            Cell: Cell in the given ``row`` and ``column``
        """

        _assert_index(row, "row")
        _assert_index(column, "column")

        return self.cells[row][column]

    def flat_cell(self, n: int) -> Cell:
        """Return the ``n``-th cell of the sudoku. Indexing works left-to-right,
        top-to-bottom.

        Args:
            n (int): Index of the cell.

        Returns:
            Cell: ``n``-th cell.
        """

        return self.flat_cells[n]

    def row(self, n: int) -> Row:
        """Return the ``n``-th row of the sudoku.

        Args:
            n (int): Index of the row.

        Returns:
            Row: ``n``-th row.
        """

        _assert_index(n, "row")

        return self.rows[n]

    def column(self, n: int) -> Column:
        """Return the ``n``-th column of the sudoku.

        Args:
            n (int): Index of the column.

        Returns:
            Row: ``n``-th column.
        """

        _assert_index(n, "column")

        return self.columns[n]

    def block(self, n: int) -> Block:
        """Return the ``n``-th block of the sudoku.

        Args:
            n (int): Index of the block.

        Returns:
            Row: ``n``-th block.
        """

        _assert_index(n, "block")

        return self.blocks[n]

    def groups(self, group: Literal["columns", "rows", "blocks"]) -> tuple[Group, ...]:
        """Convenience function to access the three groups of a Sudoku.

        Args:
            group (Literal[&quot;columns&quot;, &quot;rows&quot;, &quot;blocks&quot;]): Group to access.

        Raises:
            ValueError: If the Sudoku does not contain this group.

        Returns:
            tuple[Group, ...]: List of the groups.
        """

        if group == "columns":
            return self.columns

        if group == "rows":
            return self.rows

        if group == "blocks":
            return self.blocks

        raise ValueError(f'Unknown group "{group}".')

    def collapse(self, column: int, row: int, value: int) -> None:
        """Assign ``value`` to the cell in the specified ``column`` and ``row``. This
        also excludes ``value`` from neighbouring cell's candidates.

        Args:
            column (int): Column of the collapsed cell.
            row (int): Row of the collapsed cell.
            value (int): New value of the cell.
        """

        _assert_index(value, "value")

        cell = self.cell(column=column, row=row)
        cell.value = value

        for c in self.row(row):
            c.remove_candidate(value)

        for c in self.column(column):
            c.remove_candidate(value)

        for c in self.block(cell.block):
            c.remove_candidate(value)

    def collapse_cell(self, cell: Cell, value: int) -> None:
        """Convenience function to wrap ``Sudoku.collapse``. Equivalent to
        ``Sudoku.collapse(cell.column, cell.row, value)``.

        Args:
            cell (Cell): Cell which should be collapsed.
            value (int): New value of the cell.
        """

        self.collapse(column=cell.column, row=cell.row, value=value)

    @overload
    @classmethod
    def load(
        cls, presets: list[list[int]] | tuple[tuple[int, ...], ...]
    ) -> "Sudoku": ...

    @overload
    @classmethod
    def load(cls, presets: str) -> "Sudoku": ...

    @classmethod
    def load(
        cls, presets: list[list[int]] | tuple[tuple[int, ...], ...] | str
    ) -> "Sudoku":

        sudoku = cls()

        if isinstance(presets, str):
            rows = presets.splitlines()

        else:
            rows = presets

        for row in range(9):
            for column in range(9):
                c = int(rows[row][column])
                if c:
                    # Collapse cells only if the preset value is non-zero
                    sudoku.collapse(column=column, row=row, value=c)

        return sudoku

    def completed(self) -> bool:
        """
        Returns:
            bool: ``True`` if all cells are filled in, ``False`` otherwise.
        """

        for c in self.flat_cells:
            # Cell.value is initialized to 0
            if not c.value:
                return False

        return True

    def __str__(self) -> str:
        """Return a formatted representation of the sudoku."""

        # line types
        lt: dict[str, str] = {
            "sh": "\u2500",  # horizontal single line
            "dh": "\u2550",  # horizontal double line
            "sv": "\u2502",  # vertical single line
            "dv": "\u2551",  # vertical double line
            "ctl": "\u2554",  # corner top left
            "ctr": "\u2557",  # corner top right
            "cbl": "\u255a",  # corner bottom left
            "cbr": "\u255d",  # corner bottom right
            "tls": "\u255f",  # t-junction left single
            "tld": "\u2560",  # t-junction left double
            "trs": "\u2562",  # t-junction right single
            "trd": "\u2563",  # t-junction right double
            "tts": "\u2564",  # t-junction top single
            "ttd": "\u2566",  # t-junction top double
            "tbs": "\u2567",  # t-junction bottom single
            "tbd": "\u2569",  # t-junction bottom double
            "jss": "\u253c",  # 4-way junction single (vert), single (horiz)
            "jsd": "\u256a",  # 4-way junction single (vert), double (horiz)
            "jds": "\u256b",  # 4-way junction double (vert), single (horiz)
            "jdd": "\u256c",  # 4-way junction double (vert), double (horiz)
        }

        spacing = 2  # 0, 1
        width = 4 * spacing + 1

        # fmt: off
        # top border segment
        top_border_seg = 2 * (width * lt["dh"] + lt["tts"]) + width * lt["dh"]
        top_border = lt["ctl"] + 2 * (top_border_seg + lt["ttd"]) + top_border_seg + lt["ctr"] + "\n"

        # bottom border segment
        bottom_border_seg = 2 * (width * lt["dh"] + lt["tbs"]) + width * lt["dh"]
        bottom_border = lt["cbl"] + 2 * (bottom_border_seg + lt["tbd"]) + bottom_border_seg + lt["cbr"]

        # cell separator
        cell_sep_seg = 2 * (width * lt["sh"] + lt["jss"]) + width * lt["sh"]
        cell_sep = lt["tls"] + 2 * (cell_sep_seg + lt["jds"]) + cell_sep_seg + lt["trs"] + "\n"

        # block separator
        block_sep_seg = 2 * (width * lt["dh"] + lt["jsd"]) + width * lt["dh"]
        block_sep = lt["tld"] + 2 * (block_sep_seg + lt["jdd"]) + block_sep_seg + lt["trd"] + "\n"
        # fmt: on

        formats: list[list[tuple[str, str, str]]] = []

        for row in self.rows:
            formats.append([])
            for cell in row.cells:
                formats[-1].append(cell.format(spacing))

        # print(formats[5][1])
        # print(self.row(1).cells[5])

        output = top_border

        for cell_row in range(9):
            for format_row in range(3):
                output += lt["dv"]
                for cell_column in range(9):
                    output += formats[cell_row][cell_column][format_row]
                    output += lt["sv"] if (cell_column + 1) % 3 else lt["dv"]

                output += "\n"

            if cell_row == 8:
                output += bottom_border
            elif (cell_row + 1) % 3:
                output += cell_sep
            else:
                output += block_sep

        return output


class Solver:
    def __init__(self, sudoku: Sudoku, *, log_file: PathLike | None = None) -> None:
        self.sudoku = sudoku

        self.logging = False
        if log_file:
            self.logging = True
            self.logger = logging.getLogger("Sudoku_Solver")
            self.logger.setLevel(logging.INFO)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.INFO)
            self.logger.addHandler(stream_handler)

    def solve(self, *, interupt: bool = False, print_each: bool = False):
        while True:
            if print_each:
                print(self.sudoku)

            if self.step():
                # Sudoku was solved
                break

            if interupt:
                input("continue")

    def step(self) -> bool:

        if self.obvious_single():
            return False

        if self.hidden_single():
            return False

        if self.sudoku.completed():
            return True

        raise RuntimeError("The solver could not figure out the next step.")

    def obvious_single(self) -> bool:
        """Situation: A cell only has a single remaining candidate.

        Rule: This cell must contain the candidate as it is the only possible
        option.

        Returns:
            bool: Whether an obvious single was found.
        """

        for cell in self.sudoku.flat_cells:
            if len(cell.candidates) == 1:
                candidate = cell.candidates[0]
                self.sudoku.collapse_cell(cell=cell, value=candidate)
                self.log(
                    f"Obvious single {candidate} at (c, r) = ({cell.column}, {cell.row})"
                )
                return True

        return False

    def hidden_single(self) -> bool:
        """Situation: In a group there is only one cell that can contain a
        specific candidate.

        Rule: This cell must contain the candidate as it is the only possible
        option.

        Returns:
            bool: Whether a hidden single was found.
        """

        for candidate in range(1, 10):
            for group in ["blocks", "columns", "rows"]:
                groups = self.sudoku.groups(group=group)  # type:ignore
                for g in groups:
                    filtered = g.filter(candidate)
                    if len(filtered) == 1:
                        cell = filtered[0]
                        self.sudoku.collapse_cell(cell=cell, value=candidate)

                        self.log(
                            f"{group.capitalize()}-wise hidden single {candidate} at (c, r) = ({cell.column}, {cell.row})"
                        )

                        return True

        return False

    def log(self, msg: str, level: int = logging.INFO) -> None:
        if self.logging:
            self.logger.log(level=level, msg=msg)


if __name__ == "__main__":

    # sud = Sudoku()
    sud = Sudoku.load(
        (
            (0, 0, 0, 1, 0, 5, 7, 0, 0),
            (0, 0, 3, 0, 6, 0, 4, 0, 5),
            (1, 6, 5, 0, 0, 4, 9, 0, 0),
            (0, 0, 0, 0, 9, 6, 0, 0, 7),
            (0, 0, 7, 2, 0, 8, 3, 9, 6),
            (0, 0, 1, 7, 5, 3, 0, 8, 0),
            (0, 1, 0, 0, 2, 0, 5, 0, 0),
            (4, 8, 0, 0, 3, 0, 0, 7, 2),
            (7, 0, 0, 0, 0, 9, 0, 3, 1),
        ),
    )

    # print(sud)

    # sud.collapse(1, 1, 5)
    # sud.collapse(5, 1, 8)

    solver = Solver(sud, log_file=Path("easy.log"))
    solver.solve(interupt=True, print_each=True)
