"""Merge heads: 1a31ce608336 and 20251211_add_interaction_table

Revision ID: 20251211_merge_heads
Revises: 1a31ce608336, 20251211_add_interaction_table
Create Date: 2025-12-11 00:10:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "20251211_merge_heads"
down_revision = ("1a31ce608336", "20251211_add_interaction_table")
branch_labels = None
depends_on = None


def upgrade():
    # Merge migration: no schema changes. This revision unites two heads.
    pass


def downgrade():
    # No-op downgrade
    pass
