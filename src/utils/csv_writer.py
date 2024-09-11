import logging
import os
from csv import DictWriter
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CSVWriter:
    def __init__(self, filename: str, columns: List[str], write_interval: int = 5, csv_writing_enabled: bool = True):
        """
        A CSV writer that writes rows to disk. It writes the rows to disk in batches of write_interval.
        @param filename: The filename to write the CSV to.
        @param columns: The columns of the CSV.
        @param write_interval: The number of rows to write to disk. Improves performance thanks to buffering.
        """
        self.filename = filename
        self.csv_writing_enabled = csv_writing_enabled
        logger.info(f"CSV writing enabled status: {self.csv_writing_enabled}.")
        # Delete filename if exists
        if self.csv_writing_enabled:
            if os.path.exists(filename):
                logger.info(f"Deleting {filename} as it already exists. Creating bak file.")
                exit_code = os.system(f"mv {filename} {filename}.bak")
                if exit_code != 0:
                    logger.warning(f"Failed to create bak file for {filename}. Exit code {exit_code}.")

            # create dirs if not exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.columns = columns
        self.rows: List[Dict[str, Any]] = []
        self.header_written = False
        self.write_interval = write_interval

    def add_row(self, row: Dict[str, Any]):
        """
        Add a row to the CSV. If the number of rows is greater than or equal to write_interval, write to disk.
        Row is a dictionary with keys as column names and values as the values for the row. Usually [step, reward, seed].
        @param row: A dictionary with keys as column names and values as the values for the row.
        """
        if self.csv_writing_enabled:
            self.rows.append(row)
            if len(self.rows) >= self.write_interval:
                self.write()

    def write(self):
        """
        Write the rows to disk. Remember to call at the end of the training loop to write the remaining rows.
        """
        if self.csv_writing_enabled:
            logger.info(f"Writing {len(self.rows)} rows to {self.filename}.")
            with open(self.filename, "a") as f:
                writer = DictWriter(f, fieldnames=self.columns)
                if not self.header_written:
                    writer.writeheader()
                    self.header_written = True
                writer.writerows(self.rows)
                self.rows = []
