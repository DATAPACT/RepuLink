"""Add rating table

Revision ID: 20251211_add_rating_table
Revises: 20251211_merge_heads
Create Date: 2025-12-11 00:20:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import sqlmodel.sql.sqltypes

# revision identifiers, used by Alembic.
revision = "20251211_add_rating_table"
down_revision = "20251211_merge_heads"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "rating",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("interaction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rater_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("comment", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=False), nullable=False),
        sa.ForeignKeyConstraint(["interaction_id"], ["interaction.id"]),
        sa.ForeignKeyConstraint(["rater_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    op.drop_table("rating")
