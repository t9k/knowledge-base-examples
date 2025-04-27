#!/usr/bin/env python3
"""
Script to maintain release history for blue-green deployment.

This script creates/updates release_history.txt with records of current and deleted releases.
It adds the current database as a new release and maintains a history of previous releases.
Older releases beyond the configured limit are moved to a "deleted" section and actually deleted from Milvus.

Environment variables:
- DATABASE_NAME: Name of the database to add to the release history
- HISTORY_RELEASES_TO_KEEP: Maximum number of releases to keep in the current list
- MILVUS_URI: Milvus connection URI (required for database deletion)
- MILVUS_TOKEN: Milvus authentication token
"""

import os
import re
import datetime
import logging
from pymilvus import MilvusClient

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
MILVUS_URI = os.environ.get("MILVUS_URI")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
HISTORY_RELEASES_TO_KEEP = int(os.environ.get("HISTORY_RELEASES_TO_KEEP", "2"))

# Constants
HISTORY_FILE = "/workspace/release_history.txt"


def extract_database_name(release_entry):
    """Extract database name from a release entry.
    
    Args:
        release_entry: A release entry line from the history file
        
    Returns:
        The database name extracted from the entry
    """
    # Format is "- database_name (timestamp)"
    match = re.match(r'^- ([^ ]+) \(.*\)$', release_entry)
    if match:
        return match.group(1)
    return None


def update_release_history(database_name, max_releases, milvus_client):
    """Update the release history file with the new release.
    
    This function handles the entire workflow:
    1. Creates the history file if it doesn't exist
    2. Reads and parses the current content
    3. Adds the new release
    4. Manages the release history (moving oldest if needed)
    5. Updates the history file
    6. Actually deletes databases that were moved to deleted section
    
    Args:
        database_name: Name of the database to add
        max_releases: Maximum number of releases to keep
        milvus_client: MilvusClient instance for database operations
    """
    # Create history file if it doesn't exist
    if not os.path.exists(HISTORY_FILE):
        logger.info("Creating new release history file")
        with open(HISTORY_FILE, "w") as f:
            f.write("# Release History\n\n")
            f.write("## Current Releases\n\n")
            f.write("## Deleted Releases\n\n")
        # Read the empty file we just created
        with open(HISTORY_FILE, "r") as f:
            content = f.read()
    else:
        # Read existing content
        with open(HISTORY_FILE, "r") as f:
            content = f.read()

    # Extract sections using regex
    header_match = re.search(r'^# Release History.*?(?=\n## Current)', content,
                             re.DOTALL)
    header = header_match.group(0) if header_match else "# Release History\n"

    current_releases_match = re.search(
        r'## Current Releases\n(.*?)(?=\n## Deleted Releases)', content,
        re.DOTALL)
    current_releases = current_releases_match.group(
        1).strip() if current_releases_match else ""

    deleted_releases_match = re.search(r'## Deleted Releases\n(.*?)$', content,
                                       re.DOTALL)
    deleted_releases = deleted_releases_match.group(
        1).strip() if deleted_releases_match else ""

    # Add new release to current releases list (with timestamp)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_release = f"- {database_name} ({timestamp})"

    # Create new current releases section (with new release at the top)
    if current_releases:
        new_current_releases = f"{new_release}\n{current_releases}"
    else:
        new_current_releases = f"{new_release}"

    # Count current releases (excluding empty lines)
    release_count = len(re.findall(r'^- ', new_current_releases, re.MULTILINE))

    # Track databases that need to be deleted
    databases_to_delete = []

    # Check if we need to move the oldest release to deleted releases
    if release_count > max_releases:
        logger.info(
            f"Number of releases ({release_count}) exceeds limit ({max_releases}), moving oldest release to deleted list"
        )

        # Get the oldest release (last one in the list)
        releases = re.findall(r'^- .*$', new_current_releases, re.MULTILINE)
        oldest_release = releases[-1] if releases else None

        if oldest_release:
            # Extract database name for deletion
            db_to_delete = extract_database_name(oldest_release)
            if db_to_delete:
                databases_to_delete.append(db_to_delete)
                logger.info(f"Marking database '{db_to_delete}' for deletion")

            # Remove the oldest release from current releases
            new_current_releases = re.sub(
                re.escape(oldest_release) + r'\n?', '', new_current_releases)

            # Add the oldest release to the top of deleted releases
            if deleted_releases:
                new_deleted_releases = f"{oldest_release}\n{deleted_releases}"
            else:
                new_deleted_releases = f"{oldest_release}"
        else:
            new_deleted_releases = deleted_releases
    else:
        new_deleted_releases = deleted_releases

    # Combine all sections and write back to the file
    with open(HISTORY_FILE, "w") as f:
        f.write(
            f"{header}\n## Current Releases\n\n{new_current_releases}\n\n## Deleted Releases\n\n{new_deleted_releases}\n"
        )

    # Delete databases that were moved to deleted section
    for db_name in databases_to_delete:
        try:
            logger.info(f"Deleting database '{db_name}' from Milvus")
            if milvus_client:
                # Check if database exists before trying to delete
                existing_dbs = milvus_client.list_databases()
                if db_name in existing_dbs:
                    milvus_client.using_database(db_name)
                    for collection in milvus_client.list_collections():
                        milvus_client.drop_collection(collection)
                    milvus_client.drop_database(db_name)
                    logger.info(f"Successfully deleted database '{db_name}'")
                else:
                    logger.warning(
                        f"Database '{db_name}' not found in Milvus, could not delete"
                    )
            else:
                logger.warning(
                    "Milvus client not available, skipping actual database deletion"
                )
        except Exception as e:
            logger.error(f"Error deleting database '{db_name}': {e}")

    # Log updated content
    with open(HISTORY_FILE, "r") as f:
        history_content = f.read()
    logger.info("Release history updated successfully")
    logger.info("----- Release History -----")
    logger.info(f"\n{history_content}")


def main():
    """Main function to run the release history management workflow."""
    if not DATABASE_NAME:
        logger.error("DATABASE_NAME is required but not provided")
        raise ValueError("DATABASE_NAME environment variable is required")

    logger.info(f"Maintaining release history for database: {DATABASE_NAME}")
    logger.info(f"Maximum releases to keep: {HISTORY_RELEASES_TO_KEEP}")

    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    # Update the release history file
    update_release_history(DATABASE_NAME, HISTORY_RELEASES_TO_KEEP,
                           milvus_client)


if __name__ == "__main__":
    main()
